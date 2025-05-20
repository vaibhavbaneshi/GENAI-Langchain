from crewai import Crew, Process
from agents import blog_researcher, blog_writer
from task import research_task, write_task

#Forming the tech-focused crew with some enhanced configurations

crew=Crew(
 agents=[blog_researcher, blog_writer],
 tasks={research_task, write_task},
 process=Process.sequential,
 memory=True,
 cache=True,
 max_rpm=100,
 share_crew=True
)

##start the tash execution process the enhanced feedback
result=crew.kickoff(inputs={'topic':'Why Trump’s move to sell AMRAAM to Turkey alarms India? Ankit Agrawal Study IQ'})

print(result)