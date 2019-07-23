import pandas as pd
import numpy as np
import scipy.stats as spst
import math

class essential_utilization:

    # dwo function
    def essential_utilization(self, data_, building_floor, assigned_HC_only, interval=30,PRINTING=False):


        """
        This dwo function is written based on DWO2.1.1 https://rifiniti.app.box.com/files/0/f/2008847845/1/f_25613380221, original R code https://github.com/rifiniti/data_science/blob/master/production_code/dwo.R
        Every time we will only process data for one campus or one timezone

        :param data_: pandas DataFrame, for one campus or one timezone, contains:
            -- timestamp: numeric
            -- badge_id: character
            -- office_building_floor_id: numeric
            -- office_building_id: numeric
            -- assigned_building_id: numeric
            -- department_id: numeric
        :param building_floor: pandas DataFrame contains mapping between building and floors in the campus or the timezone. two columns: office_building_id, office_building_floor_id
        :param assigned_HC_only: boolen, TRUE or FALSE
        :param interval: numeric, is alwasy 30 mins for current Optimo
        :return: a pandas DataFrame which return utilization assigned people, assigned in campus, and people from outside utilization for every floor, every department and every 30 mins
        """

        # variable to store final utilization result
        OCCUP = pd.DataFrame()

        # add dates to the column and generate unique dates, we will process day by day
        data_['date'] = data_['timestamp'] // (24*3600)
        dates = np.unique(data_.date)

        # create a list of all buildings
        all_buildings = np.unique(building_floor['office_building_id'])

        # start to process day by day
        for date in dates:

            if PRINTING:
                print ("start processing date %s" % date)

            # selected day's data
            dat_ = data_[data_.date == date]

            # first and last timestamp of the day
            first_ts = date * 3600 * 24
            last_ts = first_ts + 60*60*24-1

            # create working-end time for each building according to average number of observations for each badge_id that has ever shown for the building in ith day
            x = [dat_.badge_id[dat_.office_building_id == b].value_counts().mean() for b in all_buildings.tolist() ]  # average number of observations for each building
            y = [0.9 if xx>=4 else -0.008333 * xx + 0.9333333 for xx in x]  # percentage used to calculate end time for each building
            end_b = [ dat_.timestamp[dat_.office_building_id == all_buildings[k]].quantile(q=y[k]) if np.isnan(y[k])==False else float('nan') for k in range(len(y))] # end time
            end_mean = np.nanmean(end_b)
            end_b = [ end_mean if np.isnan(b) else b for b in end_b ]        # for building without any data (mostly because of missing data), use the average end time of other buildings
            end_b = [ (b - first_ts)/3600 for b in end_b]

            # # variable to store utilization for selected day
            Occup = pd.DataFrame()

            # departments that exit in selected day
            departments = np.unique(dat_['department_id'])
            departments = departments[departments>0]

            # all floors that people go to
            #floors_to = np.unique(dat_.office_building_floor_id)

            # loop over each department
            for department in departments:

                if PRINTING:
                    print( "    processing department %s"  % department)

                # data for selected department
                badgeData = dat_[dat_.department_id == department]

                # only include floor that a department visit but not all floors for this department's utilization
                floors_to = np.unique(badgeData.office_building_floor_id)

                # create structure to store utilization
                time_interval = [  r * interval * 60 + first_ts for r in range(48) ]
                Time_Interval = time_interval * len(floors_to)

                occup = pd.DataFrame()
                occup['office_building_floor_id'] = np.repeat(floors_to, 48)
                occup['timestamp'] = Time_Interval
                occup['occup_assigned'] = 0.0
                occup['occup_in_campus'] = 0.0
                occup['occup_off_campus'] = 0.0

                ## start with processing utilization for people who are assgined to the buildings being studied
                badgeData1 = badgeData[badgeData.assigned_building_id.isin(all_buildings)]

                # check if there is data for this department, will skip otherwise
                if len(badgeData1):

                    ids1 = np.unique(badgeData1.badge_id)

                    # loop over each badge id
                    for id in ids1:

                        ## get data for selected badge id
                        dat = badgeData1[badgeData1.badge_id==id]

                        # remove rows that are not helpful in determine where people are
                        if len(dat)>2:
                            same_floor = np.delete(np.diff(dat.office_building_floor_id), -1)
                            row_delete = [n+1 for n in np.where(same_floor==0)]
                            if len(row_delete):
                                dat = dat.drop(dat.index[row_delete])

                        ## get timestamps and visited places
                        timestamps = dat.timestamp.tolist() # all timestamp
                        first_in = (timestamps[0] - first_ts)/float(3600) # first badge in
                        last_in = (timestamps[len(timestamps)-1] - first_ts)/float(3600) # last badge in
                        assigned_building = dat.assigned_building_id.tolist()[0] # assigned building
                        floors_visit = dat.office_building_floor_id.tolist() # floors corresponding to timestamps
                        buildings_visit = dat.office_building_id.tolist() # buildings corresponding to timestamps

                        end_time = end_b[np.where(all_buildings == assigned_building)[0][0]]

                        # geneate stay time. if first in is before building end time, using truncated normal distribution, otherwise uniform distribution
                        if first_in <= end_time:
                            mu = end_time - first_in
                            sigma = 1.5
                            #stay = last_in - first_in + 2
                            stay = spst.truncnorm(a=(last_in - first_in - mu)/sigma, b=(100 - first_in - mu)/sigma, loc=mu, scale=sigma).rvs()
                        else:
                            min_value = last_in - first_in
                            max_value = 24 - first_in
                            stay = spst.uniform(loc = min_value, scale = max_value - min_value).rvs()
                            #stay = last_in - first_in + 2

                        # working end time for this badge_id
                        end = min([first_ts + (first_in + stay) * 3600, last_ts])

                        # complete the useful timestamps and find what intervals they belong to
                        timestamps.append(end)
                        timestamps_interval = [ (t - first_ts)/(interval * 60.0) for t in timestamps  ]
                        timestamps_interval_loc = [ int(math.floor(t)) for t in timestamps_interval]

                        # loop for each small utlization pieces to calculate
                        for k in range(len(timestamps)-1):

                            if (assigned_building != buildings_visit[k]) & assigned_HC_only:
                                continue

                            ## location for the k the badge-in
                            location  = np.where(floors_to == floors_visit[k])[0].tolist()[0]

                            # calculate kth piece of utilization
                            temp_occup = [0] * 48

                            if timestamps_interval_loc[k] == timestamps_interval_loc[k+1]:

                                temp_occup[ timestamps_interval_loc[k] ] = (timestamps[k+1] - timestamps[k])/(interval*60.0)

                            elif timestamps_interval_loc[k]+1 == timestamps_interval_loc[k+1]:

                                temp_occup[ timestamps_interval_loc[k] ] = (time_interval[ timestamps_interval_loc[k]+1 ]-timestamps[k])/(interval*60.0)
                                temp_occup[ timestamps_interval_loc[k+1] ] = (timestamps[k+1]-time_interval[ timestamps_interval_loc[k+1] ])/(interval*60.0)

                            else:

                                temp_occup[ (timestamps_interval_loc[k]+1):(timestamps_interval_loc[k+1]) ] = [1] * len(range( (timestamps_interval_loc[k]+1), (timestamps_interval_loc[k+1])))
                                temp_occup[ timestamps_interval_loc[k] ] = (time_interval[ timestamps_interval_loc[k]+1 ]-timestamps[k])/(interval*60.0)
                                temp_occup[ timestamps_interval_loc[k+1] ] = (timestamps[k+1]-time_interval[timestamps_interval_loc[k+1]])/(interval*60.0)

                            change_loc = range( (location)*48, ((location+1)*48))

                            # if badge id belongs to another building, add utilization to occup_in_campus; otherwise add to occup_assigned
                            if assigned_building!=buildings_visit[k]:
                                #for l in range(len(temp_occup)):
                                occup.loc[change_loc, 'occup_in_campus'] = occup.loc[change_loc, 'occup_in_campus'] +temp_occup
                            else:
                                #for l in range(len(temp_occup)):
                                occup.loc[change_loc, 'occup_assigned'] = occup.loc[change_loc, 'occup_assigned']+temp_occup

                # if we want utilizztion for all people
                if assigned_HC_only == False:

                    badgeData2 = badgeData[ badgeData.assigned_building_id.isin(all_buildings)==False  ]

                    if len(badgeData2):

                        ids2 = np.unique(badgeData2.badge_id).tolist()

                        # loop over each badge id that is not assgined the buidlings
                        for id in ids2:

                            dat = badgeData2[ badgeData2.badge_id == id ]
                            floors_visit = dat.office_building_floor_id.tolist()

                            # get all timestamps and their differences
                            timestamps = dat.timestamp.tolist()
                            time_diff = np.diff(timestamps)/3600.0

                            # geneate random stay time for each visit, but use min(stay, next_timestamp) as visit's end
                            stay = spst.expon(scale = 1.5).rvs(size = len(timestamps)).tolist()
                            #stay = [1] * len(timestamps)
                            stay1 = stay[0:(len(stay)-1)]
                            stay2 = stay[-1]

                            stay_change_loc = np.where(stay1 > time_diff)[0]
                            for loc in stay_change_loc:
                                stay1[loc] = time_diff[loc]
                            end = min( timestamps[-1]+stay2*3600, last_ts )
                            stay2 = (end - timestamps[-1]) / 3600.0
                            stay1.append(stay2)
                            stay = stay1

                            # end of each visit
                            timestamp_start = timestamps
                            timestamp_end = [timestamp_start[l] + stay[l]*3600 for l in range(len(stay))]

                            timestamp_start_interval = [ (t-first_ts)/(interval * 60.0) for t in timestamp_start ]
                            timestamp_end_interval = [ (t-first_ts)/(interval * 60.0) for t in timestamp_end]

                            timestamp_start_interval_loc = [int(math.floor(t)) for t in timestamp_start_interval]
                            timestamp_end_interval_loc = [int(math.floor(t)) for t in timestamp_end_interval]

                            ## loop for each two start & end timestamps
                            for k in range(len(timestamp_start)):

                                location  = np.where(floors_to == floors_visit[k])[0].tolist()[0]
                                temp_occup = [0] * 48

                                if timestamp_start_interval_loc[k] == timestamp_end_interval_loc[k]:

                                    temp_occup[ timestamp_start_interval_loc[k] ] = timestamp_end_interval[k] - timestamp_start_interval[k]

                                elif timestamp_start_interval_loc[k]+1 == timestamp_end_interval_loc[k]:

                                    temp_occup[ timestamp_start_interval_loc[k] ] = 1 - (timestamp_start_interval[k] - timestamp_start_interval_loc[k] )
                                    temp_occup[ timestamp_end_interval_loc[k] ] = timestamp_end_interval[k] - timestamp_end_interval_loc[k]

                                else:

                                    temp_occup[(timestamp_start_interval_loc[k]+1):(timestamp_end_interval_loc[k])] = [1] * len(range( (timestamp_start_interval_loc[k]+1), (timestamp_end_interval_loc[k])))
                                    temp_occup[ timestamp_start_interval_loc[k] ] = 1 - (timestamp_start_interval[k] - timestamp_start_interval_loc[k] )
                                    temp_occup[ timestamp_end_interval_loc[k] ] = timestamp_end_interval[k] - timestamp_end_interval_loc[k]

                                change_loc = range( (location)*48, ((location+1)*48))
                                #for l in range(len(temp_occup)):
                                occup.loc[change_loc, 'occup_off_campus']  = occup.loc[change_loc, 'occup_off_campus']+temp_occup


                occup['department_id'] = department
                Occup = Occup.append(occup)

            OCCUP = OCCUP.append(Occup)

        return OCCUP
