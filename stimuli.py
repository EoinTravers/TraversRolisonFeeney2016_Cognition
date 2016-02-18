# -*- coding: utf-8 -*-

q_conflict1 = "A bat and a ball together costs £1.10.\n\
A bat costs £1 more than a ball.\n\
How much does a ball cost?"
resps_conflict1 = ["5p", "10p", "15p", "90p"]

q_conflict2 = "It takes 5 machines 5 minutes to make 5 \n\
widgets. How many minutes would it take 100\n\
machines to make 100 widgets?"
resps_conflict2 = ["5", "100", "50", "10"]

q_conflict3 = "In a lake, there is a patch of lily pads.\n\
Every day, the patch doubles in size.\n\
If it takes 48 days for the patch to cover\n\
the entire lake, how many days would it take\n\
for the patch to cover half of the lake?"
resps_conflict3 = ["47", "24", "12", "2"]

q_conflict4 = "If you flipped a fair coin twice, what is\n\
the probability that it would land\n\
'Heads' at least once?"
resps_conflict4 = ["75%", "50%", "25%", "100%"]

q_conflict5 = "If 3 elves can wrap 3 toys in\n\
1 hour, how many elves are needed\n\
to wrap 6 toys in 2 hours?"
resps_conflict5 = ["3", "6", "1", "12"]

q_conflict6 = "Ellen and Kim are running around a track.\n\
They run equally fast but Ellen started later.\n\
When Ellen has run 5 laps, Kim has run 10 laps.\n\
When Ellen has run 10 laps, how many has Kim run?"
resps_conflict6 = ["15", "20", "5", "19"]

q_conflict7 = "Jerry received both the 15th highest and\n\
the 15th lowest mark in the class. How many\n\
students are there in the class?"
resps_conflict7 = ["29", "30", "40", "5"]

q_conflict8 = "In an athletics team tall members tend to win\n\
three times as many medals than short members.\n\
This year the team has won 60 medals so far.\n\
How many of these have been won by short athletes?"
resps_conflict8 = ["15", "20", "30", "50"]

q_control1 = "A bat and a ball together costs £1.05.\n\
A bat costs £1.\n\
How much does a ball cost?"
resps_control1 = ["5p", "10p", "15p", "90p"]

q_control2 = "It takes a machine 5 minutes to make 5 \n\
widgets. How many minutes would it take the\n\
machines to make 100 widgets?"
resps_control2 = ["100", "5", "50", "10"]

q_control3 = "In a lake, there is a patch of lily pads.\n\
Every day, the patch grows by 10m².\n\
If it takes 48 days for the patch to cover\n\
the 150m², how many days would it take\n\
for the patch to cover 140m²?"
resps_control3 = ["47", "24", "12", "2"]

q_control4 = "If you flipped a fair coin twice, what is\n\
the probability that it would land\n\
'Heads' exactly once?"
resps_control4 = ["50%", "25%", "75%", "100%"]

q_control5 = "If 3 elves can wrap 3 toys in\n\
1 hour, how many toys could 6 elves\n\
wrap in half an hour?"
resps_control5 = ["3", "6", "1", "12"]

q_control6 = "Ellen and Kim are running around a track.\n\
They started at the same time, but Kim is twice as fast as Ellen.\n\
When Ellen has run 5 laps, Kim has run 10 laps.\n\
When Ellen has run 10 laps, how many has Kim run?"
resps_control6 = ["20", "15", "5", "19"]

q_control7 = "Jerry received both the 2nd highest and\n\
the 2nd lowest mark in the class. How many\n\
students are there in the class?"
resps_control7 = ["3", "2", "5", "10"]

q_control8 = "In an athletics team tall members tend to win\n\
twice as many medals than short members.\n\
This year the team has won 60 medals so far.\n\
How many of these have been won by short athletes?"
resps_control8 = ["20", "15", "30", "50"]

conflict_qs = [q_conflict1, q_conflict2, q_conflict3, q_conflict4,
				q_conflict5, q_conflict6, q_conflict7, q_conflict8]
control_qs = [q_control1, q_control2, q_control3, q_control4,
				q_control5, q_control6, q_control7, q_control8]
conflict_resps = [resps_conflict1, resps_conflict2, resps_conflict3, resps_conflict4,
				resps_conflict5, resps_conflict6, resps_conflict7, resps_conflict8]
control_resps = [resps_control1, resps_control2, resps_control3, resps_control4,
				resps_control5, resps_control6, resps_control7, resps_control8]
				
question_listA = [	q_conflict1, q_control2, q_conflict3, q_control4,
						q_conflict5, q_control6, q_conflict7, q_control8]
question_listB = [	q_control1, q_conflict2, q_control3, q_conflict4,
						q_control5, q_conflict6, q_control7, q_conflict8]


response_listA = [	resps_conflict1, resps_control2, resps_conflict3, resps_control4,
						resps_conflict5, resps_control6, resps_conflict7, resps_control8]
response_listB = [	resps_control1, resps_conflict2, resps_control3, resps_conflict4,
						resps_control5, resps_conflict6, resps_control7, resps_conflict8]
					
code_listA = ['C1', 'B2', 'C3', 'B4', 'C5', 'B6', 'C7', 'B8']
code_listB = ['B1', 'C2', 'B3', 'C4', 'B5', 'C6', 'B7', 'C8']
