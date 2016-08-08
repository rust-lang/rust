- Feature Name: north_star
- Start Date: 2016-08-07
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

A refinement of the Rust planning and reporting process, to establish a shared
vision of the language we are building toward among contributors, to make clear
the roadmap toward that vision, and to celebrate our achievements.

Rust's roadmap will be established in year-long cycles, where we identify up
front - together, as a project - the most critical problems facing the language,
along with the story we want to be able to tell the world about Rust. Work
toward solving those problems, our short-term goals, will be decided in
quarter-long cycles by individual teams. Goals that result in stable features
will be assigned to release milestones for the purposes of reporting the project
roadmap.

At the end of the year we will deliver a public facing retrospective, describing
the goals we achieved and how to use the new features in detail. It will
celebrate the year's progress in The Rust Project toward our goals, as well as
achievements in the wider community. It will celebrate our performance and
anticipate its impact on the coming year.

The primary outcome for these changes to the process are that we will have a
consistent way to:

- Decide our project-wide goals through consensus.
- Advertise our goals as a published roadmap.
- Celebrate our achievements with an informative publicity-bomb.

# Motivation
[motivation]: #motivation

Rust is a massive system, developed by a massive team of mostly-independent
contributors. What we've achieved together already is mind-blowing: we've
created a uniquely powerful platform that solves problems that the computing
world had nearly given up on, and jumpstarted a new era in systems
programming. Now that Rust is out in the world, proving itself to be a stable
foundation for building the next generation of computing systems, the
possibilities open to us are nearly endless.

And that's a big problem.

For many months approaching the release of Rust 1.0 we had a clear, singular
goal: get Rust done and deliver it to the world. We knew precisely the discreet
steps necessary to get there, and although it was a tense period where the
entire future of the project was on the line, we were united in a single
mission. As The Rust Project Developers we were pumped up, and our user base -
along with the wider programming world - were excited to see what we would
deliver.

The same has not been true since. We've had a number of major goals - refactor
the compiler, enable strong IDE support, make cross-compilation easier, increase
community diversity - but it's not clear that we've been as focused on them as
needed. Even where there are clear strategic priorities in the project, they are
often under-emphasized in the way we talk about Rust, under-prioritized when we
do our own work or in our efforts to rally contributions, under-staffed by both
Mozilla and community contributors, and backburnered in favor of more present
issues. We are overwhelmed by an avalanche of promising ideas, with major RFCs
demanding attention (and languishing in the queue for months), TODO another
clause to make this sentence shine.

Compounding this problem is that we have no clear end state for our efforts, no
major deliverable to show for all our work, for the community to rally behind,
and for the user base to anticipate. To a great degree this is a result of our
own successes - we have a short, time-based release cycle where new features
drip out as they become available, and a feature integration process that places
incredible emphasis on maintaining stability. It works shockingly well! But Rust
releases are boring 😢 (admitedly some of the reason for this is that language
features have been delayed waiting on internal compiler refactoring). And -
perhaps surprisingly - our rapid release process seems to cause work to proceed
slowly: the lack of deadlines for features reduces the pressure to get them
done, and today there are many approved RFCs languishing in a half-finished
state, with no-one urgently championing their completion. The slow trickle of
features reduces opportunities to make a big public 'splash' upon release,
lessening the impact of our work.

The result is that there is a lack of direction in Rust, both real and
perceieved.

This RFC proposes changes to the way The Rust Project plans its work,
communicates and monitors its progress, directs contributors to focus on the
strategic priorities of the project, and finally, delivers the results of its
effort to the world.

The changes proposed here are intended to work with the particular strengths of
our project - community development, collaboration, distributed teams, loose
management structure, constant change and uncertanty. It should introduce
minimal additional burden on Rust team members, who are already heavily
overtasked. The proposal does not attempt to solve all problems of project
management in Rust, nor to fit the Rust process into any particular project
mamnagement structure. Let's make a few incremental improvements that will have
the greatest impact, and that we can accomplish without disruptive changes to
the way we work today.

# Detailed design
[design]: #detailed-design

Rust's roadmap will be established in year-long cycles, where we identify up
front, as a project, the most critical problems facing the language, formulated
as _problem statements_. Work toward solving those problems, _goals_, will be
planned in quarter-long cycles by individual teams. _goals_ that result in
stable features will be assigned to _release milestones_ for the purposes of
reporting the project roadmap. Along the way, teams will be expected to maintain
_tracking issues_ that communicate progress toward the project's goals.

The end-of-year retrospective is a 'rallying point'. Its primary purposes are to
create anticipation of a major event in the Rust world, to motivate (rally)
contributors behind the goals we've established to get there, and generate a big
PR-bomb where we can brag to the world about what we've done. It can be thought
of as a 'state of the union'. This is where we tell Rust's story, describe the
new best practices enabled by the new features we've delivered, celebrate those
contributors who helped achieve our goals, honestly evaluate our performance,
and look forward to the year to come.

## Summary of terminology

- _problem statement_ - A description of a major issue facing Rust, possibly
  spanning multiple teams and disciplines. We decide these together every year
  so that everybody understands the direction the project is taking. These are
  used as the broad basis for decision making throughout the year.
- _goal_ - These are set by individual teams quarterly, in service of solving
  the problems identified by the project. They have estimated deadlines, and
  those that result in stable features have estimated release numbers. Goals may
  be subdivided into further discrete tasks on the issue tracker.
- _retrospective_ - At the end of the year we deliver a retrospective report. It
  presents the result of work toward each of our goals in a way that serves to
  reinforce the year's narrative. These are written for public consumption,
  showing off new features, surfacing interesting technical details, and
  celebrating those contributors who contribute to achieving the project's goals
  and resolving it's problems.
- _quarterly milestone_ - All goals have estimates for completion, placed on
  quarterly milestones. Each quarter that a goal remains incomplete it must be
  re-triaged and re-estimated by the responsible team.

## The big planning cycle (problem statements and the narrative arc)

The big cycle spans one year. At the beginning of the cycle we identify areas of
Rust that need the most improvement, and at the end of the cycle is a 'rallying
point' where we deliver to the world the results of our efforts. We choose
year-long cycles because a year is enough time to accomplish relatively large
goals; and because having the rallying point occur at the same time every year
makes it easy to know when to anticipate big news from the project.

This planning effort is _problem-oriented_. In our collective experience we have
consistently seen that spending up front effort focusing on motivation - even
when we have strong ideas about the solutions - is a critical step in building
consensus. It avoids surprises and hurt feelings, and establishes a strong causal
record for explaining decisions in the future.

At the beginning of the cycle we spend no more than one month deciding on a
small set of _problem statements_ for the project, for the year. The number
needs to be small enough to present to the community managably, while also
sufficiently motivating the primary work of all the teams for the year. 8-10 is
a reasonable guideline. This planning takes place via the RFC process and is
open to the entire community. The result of the process is the yearly 'north
star RFC'.

We strictly limit the planning phase to one month in order to keep the
discussion focused and to avoid unrestrained bikeshedding. The activities
specified here are not the focus of the project and we need to get through them
efficiently and get on with the actual work.

The core team is responsible for initiating the process, either on the internals
forum or directly on the RFC repository, and the core team is responsible for
merging the final RFC, thus it will be their responsibility to ensure that the
discussion drives to a reasonable conclusion in time for the deadline.

The problem statements established here determine the strategic direction of the
project. They identify critical areas where the project is lacking and represent
a public commitment to fixing them.

TODO: How do we talk about solutions during this process? We certainly will have
lots of ideas about how these problems are going to get solved, and we can't
pretend like they don't exist.

Problem statements consist of a single sentence summarizing the problem, and one
or more paragraph describing it in details. Examples of good problem statements
might be:

- The Rust compiler is slow
- Rust lacks world-class IDE support
- The Rust story for asynchronous I/O is incomplete
- Rust compiler errors are dificult to understand
- Plugins need to be on path to stabilization
- Rust doesn't integrate well with garbage collectors
- Inability to write truly zero-cost abstractions (due to lack of
  specialization) (TODO this is awfully goal-oriented, also not a complete
  sentence)
- We would like the Rust community to be more diverse
- It's too hard to obtain Rust for the platforms people want to target

During the actual process each of these would be accompanied by a paragraph or
more of justification.

Once the year's problem statements are decided, a metabug is created for each on
the rust-lang/rust issue tracker and tagged `R-problem-statement`. In the OP of
each metabug the teams are responsible for maintaining a list of their goals,
linking to tracking issues.

## The little planning cycle (goals and tracking progress)

TODO: This is the most important part of the RFC mechanically and needs to be
clear so teams can just read it and follow the instructions.

The little cycle is where the solutions take shape and are carried out. They
last one quarter - 3 months - and are the responsibility of individual teams.

Each cycle the teams will have one week to update their set of _goals_. This
includes both creating new goals and reviewing and revising existing goals. A
goal describes a task that contributes to solving the year's problems. It may or
may not involve a concrete deliverable, and it may be in turn subdivided into
further goals.

The social process of the quarterly planning cycle is less strict, but it
should be conducted in a way that allows open feedback. It is suggested that
teams present their quarterly plan on internals.rust-lang.org at the beginning
of the week, solicit feedback, then finalize them at the end of the week.

All goals have estimated completion dates. There is no limit on the duration of
a single goal, but they are encouraged to be scoped to less than a quarter year
of work. Goals that are expected to take more than a quarter _must_ be
subdivided into smaller goals of less than a quarter, each with their own
estimates. These estimates are used to place goals onto quarterly milestones.

Not all the work items done by teams in a quarter should be considered a goal
nor should they be. Goals only need to be granular enough to demonstrate
consistent progress toward solving the project's problems. Work that
contributors toward quarterly goals should still be tracked as sub-tasks of
those goals, but only needs to be filed on the issue tracker and not reported
directly as goals on the roadmap.

For each goal the teams will create an issue on the issue tracker tagged with
`R-goal`. Each goal must be described in a single sentence summary (TODO what
makes a good summary?). Goals with sub-goals and sub-tasks must list them in the
OP in a standard format.

During each planning period all goals must be triaged and updated for the
following information:

- The set of sub-goals and sub-tasks and their status
- The estimated date of completion for goals

## The retrospective (rallying point)

- Written for broad public consumption
- Detailed
- Progress toward goals
- Demonstration of new features
- Technical details
- Reinforce the project narrative
- Celebrate contributors who accomplished our goals
- Celebrate the evolution of the ecosystem
- Evaluation of performance, missed goals

TODO How is it constructed?

## Release estimation

The teams are responsible for estimating only the _timeframe_ in which they
complete their work, but possibly the single most important piece of information
desired by users is to know _in what release_ any given feature will become
available.

To reduce process burden on team members we will not require them to make
that estimate themselves, instead a single person will have the responsibility
each quarter to examine the roadmap, its goals and time estimates, and turn
those into release estimates for individual features.

The precise mechanics are to be determined.

## Presenting the roadmap

As a result of this process the Rust roadmap for the year is encoded in three
main ways, that evolve over the year:

- The north-star RFC, which contains the problem statements collected in one
  place
- The R-problem-statement issues, which contain the individual problem
  statements, each linking to supporting goals
- The R-goal issues, which contain the work items, tagged with metadata
  indicating their statuses.

Alone, this is perhaps sufficient for presenting the roadmap. A user could run a
GitHub query for all `R-problem-statement` issues, and by digging through them
get a reasonably accurate picture of the roadmap.

We may additionally develop tools to present this information in a more
accessible form (for a prototype see [1]).

[1]: https://brson.github.io/rust-z

## Calendar

The timing of the events specified by this RFC is precisely specified in order
to limit bikeshedding. The activities specified here are not the focus of the
project and we need to get through them efficiently and get on with the actual
work.

The north star RFC development happens during the month of September, starting
September 1 and ending by October 1. This means that an RFC must be ready for
RFC by the last week of September. We choose september for two reasons: it is
the final month of a calendar quarter, allowing the beginning of the years work
to commence at the beginning of calendar Q4; we choose Q4 because it is the
traditional conference season and allows us opportunities to talk publicly about
both our previous years progress as well as next years ambitions.

Following from the September planning month, the quarterly planning cycles take
place for exactly one week at the beginning of the calendar quarter; and the
development of the yearly retrospective approximately for the month of August.

## Summary of mechanics

There are four primary new mechanism introduced by this RFC

- North star RFC. Each year in September the entire project comes together
  to produce this. It is what drives the evolution of the project roadmap
  over the next year.
- `R-problem-statement` tag. The north star RFC defines problem statements that
  are filed and tagged on the issue tracker. The `R-problem-statement` issues in
  turn link to the goals that support them.
- `R-goal`. Reevaluated every quarter by the teams, with feedback from
  the wider community, these are filed on the issue tracker, tagged `R-goal` and
  linked to the `R-problem-statement` issue they support.
- End-of-year retrospective blog post. In the final month we write a detailed
  blog post that hypes up our amazing work.

For simplicity, all `R-problem-statement` and `R-goal` issues live in
rust-lang/rust, even when they primarily entail work on other code-bases.

# Drawbacks
[drawbacks]: #drawbacks

The yearly north star RFC could be an unpleast bikeshed. Maybe nobody actually
agrees on the project's direction.

This imposes more work on teams to organize their goals.

There is no mechanism here for presenting the roadmap.

The end-of-year retrospective will require significant effort. It's not clear
who will be motivated to do it, and at the level of quality it demands.

# Alternatives
[alternatives]: #alternatives

Instead of imposing further process structure on teams we might attempt to
derive a roadmap soley from the data they are currently producing.

To serve the purposes of a 'rallying point', a high-profile deliverable, we
might release a software product instead of the retrospective. A larger-scope
product than the existing rustc+cargo pair could accomplish this, i.e.
The Rust Platform.

Another rallying point could be a long-term support release.

# Unresolved questions
[unresolved]: #unresolved-questions

Are 1 year cycles long enough?

Does the yearly report serve the purpose of building anticipation, motivation,
and creating a compelling PR-bomb?

Is a consistent time-frame for the big cycle really the right thing?  One of the
problems we have right now is that our release cycles are so predictable they
are boring. It could be more exciting to not know exactly when the cycle is
going to end, to experience the tension of struggling to cross the finish line.

How can we account for work that is not part of the planning process
described here?

How can we avoid adding new tags?

How do we address problems that are outside the scope of the standard library
and compiler itself? Would have used 'the rust platform' and related processes.

How do we motivate the improvement of rust-lang, other libraries?

'Problem statement' is not inspiring terminology. We don't want to our roadmap
to be front-loaded with 'problems'.

Likewise, 'goal' and 'retrospective' could be more colorful.

How can we work in an inspiring 'vision statement'?

Can we call the yearly RFC the 'north start RFC'? Too many concepts?

Does the yearly planning really need to be an RFC?

Likewise, _this RFC_ is currently titled 'north-star'.

What about tracking work that is not part of R-problem-statement and R-goal.  I
originally wanted to track all features in a roadmap, but this does not account
for anything that has not been explicitly identified as supporting the
roadmap. As formulated this does not provide an easy way to find the status of
arbitrary features in the RFC pipeline.

How do we present the roadmap? Communicating what the project is working on and
toward is one of the _primary goals_ of this RFC and the solution it proposes is
minimal - read the R-problem-statement issues.
