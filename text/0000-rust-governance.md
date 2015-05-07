- Feature Name: not applicable
- Start Date: 2015-02-27
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

This RFC proposes to expand, and make more explicit, Rust's governance
structure. It seeks to supplement today's core team with several
*subteams* that are more narrowly focused on specific areas of
interest.

*Thanks to Nick Cameron, Manish Goregaokar, Yehuda Katz, Niko Matsakis and Dave
 Herman for many suggestions and discussions along the way.*

# Motivation

Rust's governance has evolved over time, perhaps most dramatically
with the introduction of the RFC system -- which has itself been
tweaked many times. RFCs have been a major boon for improving design
quality and fostering deep, productive discussion. It's something we
all take pride in.

That said, as Rust has matured, a few growing pains have emerged.

We'll start with a brief review of today's governance and process,
then discuss what needs to be improved.

## Background: today's governance structure

Rust is governed by a
[core team](https://github.com/rust-lang/rust-wiki-backup/blob/master/Note-core-team.md),
which is ultimately responsible for all decision-making in the
project. Specifically, the core team:

* Sets the overall direction and vision for the project;
* Sets the priorities and release schedule;
* Makes final decisions on RFCs.

The core team currently has 8 members, including some people working
full-time on Rust, some volunteers, and some production users.

Most technical decisions are decided through the
[RFC process](https://github.com/rust-lang/rfcs#what-the-process-is).
RFCs are submitted for essentially all changes to the language,
most changes to the standard library, and
[a few other topics](https://github.com/rust-lang/rfcs#when-you-need-to-follow-this-process).
RFCs are either closed immediately (if they are clearly not viable),
or else assigned a *shepherd* who is responsible for keeping the
discussion moving and ensuring all concerns are responded to.

The final decision to accept or reject an RFC is made by the core
team. In many cases this decision follows after many rounds of
consensus-building among all stakeholders for the RFC. In the end,
though, most decisions are about weighting various tradeoffs, and the
job of the core team is to make the final decision about such
weightings in light of the overall direction of the language.

## What needs improvement

At a high level, we need to improve:

* Process scalability.
* Stakeholder involvement.
* Clarity/transparency.
* Moderation processes.

Below, each of these bullets is expanded into a more detailed analysis
of the problems. These are the problems this RFC is trying to
solve. The "Detailed Design" section then gives the actual proposal.

### Scalability: RFC process

In some ways, the RFC process is a victim of its own success: as the
volume and depth of RFCs has increased, it's harder for the entire
core team to stay educated and involved in every RFC. The
[shepherding process](https://github.com/rust-lang/rfcs#the-role-of-the-shepherd)
has helped make sure that RFCs don't fall through the cracks, but even
there it's been hard for the relatively small number of shepherds to
keep up (on top of the other work that they do).

Part of the problem, of course, is due to the current push toward 1.0,
which has both increased RFC volume and takes up a great deal of
attention from the core team. But after 1.0 is released, the community
is likely to grow significantly, and feature requests will only
increase.

Growing the core team over time has helped, but there's a practical
limit to the number of people who are jointly making decisions and
setting direction.

A distinct problem in the other direction has also emerged recently: we've
slowly been requiring RFCs for increasingly minor changes. While it's important
that user-facing changes and commitments be vetted, the process has started to
feel heavyweight (especially for newcomers), so a recalibration may be in order.

We need a way to scale up the RFC process that:

* Ensures each RFC is thoroughly reviewed by several people with
  interest and expertise in the area, but with different perspectives
  and concerns.

* Ensures each RFC continues moving through the pipeline at a
  reasonable pace.

* Ensures that accepted RFCs are well-aligned with the values, goals,
  and direction of the project, and with other RFCs (past, present,
  and future).

* Ensures that simple, uncontentious changes can be made quickly, without undue
  process burden.

### Scalability: areas of focus

In addition, there are increasingly areas of important work that are
only loosely connected with decisions in the core language or APIs:
tooling, documentation, infrastructure, for example. These areas all
need leadership, but it's not clear that they require the same degree
of global coordination that more "core" areas do.

These areas are only going to increase in number and importance, so we
should remove obstacles holding them back.

### Stakeholder involvement

RFC shepherds are intended to reach out to "stakeholders" in an RFC,
to solicit their feedback. But that is different from the stakeholders
having a direct role in decision making.

To the extent practical, we should include a diverse range of
perspectives in both design and decision-making, and especially
include people who are most directly affected by decisions: users.

We have taken some steps in this direction by diversifying the core
team itself, but (1) members of the core team by definition need to
take a balanced, global view of things and (2) the core team should
not grow too large. So some other way of including more stakeholders
in decisions would be preferable.

### Clarity and transparency

Despite many steps toward increasing the clarity and openness of
Rust's processes, there is still room for improvement:

* The priorities and values set by the core team are not always
  clearly communicated today. This in turn can make the RFC process
  seem opaque, since RFCs move along at different speeds (or are even
  closed as postponed) according to these priorities.

  At a large scale, there should be more systematic communication
  about high-level priorities. It should be clear whether a given RFC
  topic would be considered in the near term, long term, or
  never. Recent blog posts about the 1.0 release and stabilization
  have made a big step in this direction. After 1.0, as part of the
  regular release process, we'll want to find some regular cadence for
  setting and communicating priorities.

  At a smaller scale, it is still the case that RFCs fall through the
  cracks or have unclear statuses (see Scalability problems
  above). Clearer, public tracking of the RFC pipeline would be a
  significant improvement.

* The decision-making process can still be opaque: it's not always
  clear to an RFC author exactly when and how a decision on the RFC
  will be made, and how best to work with the team for a favorable
  decision. We strive to make core team meetings as *uninteresting* as
  possible (that is, all interesting debate should happen in public
  online communication), but there is still room for being more
  explicit and public.

### Community norms and the Code of Conduct

Rust's design process and community norms are closely intertwined. The
RFC process is a joint exploration of design space and tradeoffs, and
requires consensus-building. The process -- and the Rust community --
is at its best when all participants recognize that

> ... people have differences of opinion and that every design or
> implementation choice carries a trade-off and numerous costs. There
> is seldom a right answer.

This and other important values and norms are recorded in the
[project code of conduct (CoC)](http://www.rust-lang.org/conduct.html),
which also includes language about harassment and marginalized groups.

Rust's community has long upheld a high standard of conduct, and has
earned a reputation for doing so.

However, as the community grows, as people come and go, we must
continually work to maintain this standard. Usually, it suffices to
lead by example, or to gently explain the kind of mutual respect that
Rust's community practices. Sometimes, though, that's not enough, and
explicit moderation is needed.

One problem that has emerged with the CoC is the lack of clarity about
the mechanics of moderation:

* Who is responsible for moderation?
* What about conflicts of interest? Are decision-makers also moderators?
* How are moderation decisions reached? When are they unilateral?
* When does moderation begin, and how quickly should it occur?
* Does moderation take into account past history?
* What venues does moderation apply to?

Answering these questions, and generally clarifying how the CoC is viewed and
enforced, is an important step toward scaling up the Rust community.

# Detailed design

The basic idea is to supplement the core team with several "subteams". Each
subteam is focused on a specific area, e.g., language design or libraries. Most
of the RFC review process will take place within the relevant subteam, scaling
up our ability to make decisions while involving a larger group of people in
that process.

To ensure global coordination and a strong, coherent vision for the project as a
whole, **each subteam is led by a member of the core team**.

## Subteams

**The primary roles of each subteam are**:

* Shepherding RFCs for the subteam area. As always, that means (1) ensuring that
  stakeholders are aware of the RFC, (2) working to tease out various design
  tradeoffs and alternatives, and (3) helping build consensus.

* Accepting or rejecting RFCs in the subteam area.

* Setting policy on what changes in the subteam area require RFCs, and reviewing
  direct PRs for changes that do not require an RFC.

* Delegating *reviewer rights* for the subteam area. The ability to `r+` is not
  limited to team members, and in fact earning `r+` rights is a good stepping
  stone toward team membership. Each team should set reviewing policy, manage
  reviewing rights, and ensure that reviews take place in a timely manner.
  (Thanks to Nick Cameron for this suggestion.)

Subteams make it possible to involve a larger, more diverse group in the
decision-making process. In particular, **they should involve a mix of**:

* Rust project leadership, in the form of at least one core team member (the
  leader of the subteam).

* Area experts: people who have a lot of interest and expertise in the subteam
  area, but who may be far less engaged with other areas of the project.

* Stakeholders: people who are strongly affected by decisions in the
  subteam area, but who may not be experts in the design or
  implementation of that area. *It is crucial that some people heavily
  using Rust for applications/libraries have a seat at the table, to
  make sure we are actually addressing real-world needs.*

Members should have demonstrated a good sense for design and dealing with
tradeoffs, an ability to work within a framework of consensus, and of course
sufficient knowledge about or experience with the subteam area. Leaders should
in addition have demonstrated exceptional communication, design, and people
skills. They must be able to work with a diverse group of people and help lead
it toward consensus and execution.

Each subteam is led by a member of the core team. **The leader is responsible for**:

* Setting up the subteam:

    * Deciding on the initial membership of the subteam (in consultation with
      the core team). Once the subteam is up and running.

    * Working with subteam members to determine and publish subteam policies and
      mechanics, including the way that subteam members join or leave the team
      (which should be based on subteam consensus).

* Communicating core team vision downward to the subteam.

* Alerting the core team to subteam RFCs that need global, cross-cutting
  attention, and to RFCs that have entered the "final comment period" (see below).

* Ensuring that RFCs and PRs are progressing at a reasonable rate, re-assigning
  shepherds/reviewers as needed.

* Making final decisions in cases of contentious RFCs that are unable to reach
  consensus otherwise (should be rare).

The way that subteams communicate internally and externally is left to each
subteam to decide, but:

* Technical discussion should take place as much as possible on public forums,
  ideally on RFC/PR threads and tagged discuss posts.

* Each subteam will have a dedicated
  [discuss forum](http://internals.rust-lang.org/) tag.

* Subteams should actively seek out discussion and input from stakeholders who
  are not members of the team.

* Subteams should have some kind of regular meeting or other way of making
  decisions. The content of this meeting should be summarized with the rationale
  for each decision -- and, as explained below, decisions should generally be
  about weighting a set of already-known tradeoffs, not discussing or
  discovering new rationale.

* Subteams should regularly publish the status of RFCs, PRs, and other news
  related to their area. Ideally, this would be done in part via a dashboard
  like [the Homu queue](http://buildbot.rust-lang.org/homu/queue/rust)

## Core team

**The core team serves as leadership for the Rust project as a whole**. In
  particular, it:

* **Sets the overall direction and vision for the project.** That means setting
  the core values that are used when making decisions about technical
  tradeoffs. It means steering the project toward specific use cases where Rust
  can have a major impact. It means leading the discussion, and writing RFCs
  for, *major* initiatives in the project.

* **Sets the priorities and release schedule.** Design bandwidth is limited, and
  it's dangerous to try to grow the language too quickly; the core team makes
  some difficult decisions about which areas to prioritize for new design, based
  on the core values and target use cases.

* **Focuses on broad, cross-cutting concerns.** The core team is specifically
  designed to take a *global* view of the project, to make sure the pieces are
  fitting together in a coherent way.

* **Spins up or shuts down subteams.** Over time, we may want to expand the set
  of subteams, and it may make sense to have temporary "strike teams" that focus
  on a particular, limited task.

* **Decides whether/when to ungate a feature.** While the subteams make
  decisions on RFCs, the core team is responsible for pulling the trigger that
  moves a feature from nightly to stable. This provides an extra check that
  features have adequately addressed cross-cutting concerns, that the
  implementation quality is high enough, and that language/library commitments
  are reasonable.

The core team should include both the subteam leaders, and, over time, a diverse
set of other stakeholders that are both actively involved in the Rust community,
and can speak to the needs of major Rust constituencies, to ensure that the
project is addressing real-world needs.

## Decision-making

### Consensus

Rust has long used a form of [consensus decision-making][consensus]. In a
nutshell the premise is that a successful outcome is not where one side of a
debate has "won", but rather where concerns from *all* sides have been addressed
in some way. **This emphatically does not entail design by committee, nor
compromised design**. Rather, it's a recognition that

> ... every design or implementation choice carries a trade-off and numerous
> costs. There is seldom a right answer.

Breakthrough designs sometimes end up changing the playing field by eliminating
tradeoffs altogether, but more often difficult decisions have to be made. **The
key is to have a clear vision and set of values and priorities**, which is the
core team's responsibility to set and communicate, and the subteam's
responsibility to act upon.

Whenever possible, we seek to reach consensus through discussion and design
revision. Concretely, the steps are:

* Initial RFC proposed, with initial analysis of tradeoffs.
* Comments reveal additional drawbacks, problems, or tradeoffs.
* RFC revised to address comments, often by improving the design.
* Repeat above until "major objections" are fully addressed, or it's clear that
  there is a fundamental choice to be made.

Consensus is reached when most people are left with only "minor" objections,
i.e., while they might choose the tradeoffs slightly differently they do not
feel a strong need to *actively block* the RFC from progressing.

One important question is: consensus among which people, exactly? Of course, the
broader the consensus, the better. But at the very least, **consensus within the
members of the subteam should be the norm for most decisions.** If the core team
has done its job of communicating the values and priorities, it should be
possible to fit the debate about the RFC into that framework and reach a fairly
clear outcome.

[consensus]: http://en.wikipedia.org/wiki/Consensus_decision-making

### Lack of consensus

In some cases, though, consensus cannot be reached. These cases tend to split
into two very different camps:

* "Trivial" reasons, e.g., there is not widespread agreement about naming, but
  there is consensus about the substance.

* "Deep" reasons, e.g., the design fundamentally improves one set of concerns at
  the expense of another, and people on both sides feel strongly about it.

In either case, an alternative form of decision-making is needed.

* For the "trivial" case, usually either the RFC shepherd or subteam leader will
  make an executive decision.

* For the "deep" case, the subteam leader is empowered to make a final decision,
  but should consult with the rest of the core team before doing so.

### How and when RFC decisions are made, and the "final comment period"

Each RFC has a shepherd drawn from the relevant subteam. The shepherd is
responsible for driving the consensus process -- working with both the RFC
author and the broader community to dig out problems, alternatives, and improved
design, always working to reach broader consensus.

At some point, the RFC comments will reach a kind of "steady state", where no
new tradeoffs are being discovered, and either objections have been addressed,
or it's clear that the design has fundamental downsides that need to be weighed.

At that point, the shepherd will announce that the RFC is in a "final comment
period" (which lasts for one week). This is a kind of "last call" for strong
objections to the RFC. **The announcement of the final comment period for an RFC
should be very visible**; it should be included in the subteam's periodic
communications.

> Note that the final comment period is in part intended to help keep RFCs
> moving. Historically, RFCs sometimes stall out at a point where discussion has
> died down but a decision isn't needed urgently. In this proposed model, the
> RFC author could ask the shepherd to move to the final comment period (and
> hence toward a decision).

After the final comment period, the subteam can make a decision on the RFC. The
role of the subteam at that point is *not* to reveal any new technical issues or
arguments; if these come up during discussion, they should be added as comments
to the RFC, and it should undergo another final comment period.

Instead, the subteam decision is based on **weighing the already-revealed
tradeoffs against the project's priorities and values** (which the core team is
responsible for setting, globally). In the end, these decisions are about how to
weight tradeoffs. The decision should be communicated in these terms, pointing
out the tradeoffs that were raised and explaining how they were weighted, and
**never introducing new arguments**.

## Keeping things lightweight

In addition to the "final comment period" proposed above, this RFC proposes some
further adjustments to the RFC process to keep it lightweight.

A key observation is that, thanks to the stability system and nightly/stable
distinction, **it's easy to experiment with features without commitment**.

### Clarifying what needs an RFC

Over time, we've been drifting toward requiring an RFC for essentially any
user-facing change, which sometimes means that very minor changes get stuck
awaiting an RFC decision. While subteams + final comment period should help keep
the pipeline flowing a bit better, it would also be good to allow "minor"
changes to go through without an RFC, provided there is sufficient review in
some other way. (And in the end, the core team ungates features, which ensures
at least a final review.)

This RFC does not attempt to answer the question "What needs an RFC", because
that question will vary for each subteam. However, this RFC stipulates that each
subteam should set an explicit policy about:

1. What requires an RFC for the subteam's area, and
2. What the non-RFC review process is.

These guidelines should try to keep the process lightweight for minor changes.

### Clarifying the "finality" of RFCs

While RFCs are very important, they do not represent the final state of a
design. Often new issues or improvements arise during implementation, or after
gaining some experience with a feature. **The nightly/stable distinction exists
in part to allow for such design iteration.**

Thus RFCs do not need to be "perfect" before acceptance. If consensus is reached
on major points, the minor details can be left to implementation and revision.

Later, if an implementation differs from the RFC in *substantial* ways, the
subteam should be alerted, and may ask for an explicit amendment RFC. Otherwise,
the changes should just be explained in the commit/PR.

## The teams

With all of that out of the way, what subteams should we start with? This RFC
proposes the following initial set:

* Language design
* Libraries
* Compiler
* Tooling and infrastructure
* Moderation

In the long run, we will likely also want teams for documentation and for
community events, but these can be spun up once there is a more clear need (and
available resources).

### Language design team

Focuses on the *design* of language-level features; not all team members need to
have extensive implementation experience.

Some example RFCs that fall into this area:

* [Associated types and multidispatch](https://github.com/rust-lang/rfcs/pull/195)
* [DST coercions](https://github.com/rust-lang/rfcs/pull/982)
* [Trait-based exception handling](https://github.com/rust-lang/rfcs/pull/243)
* [Rebalancing coherence](https://github.com/rust-lang/rfcs/pull/1023)
* [Integer overflow](https://github.com/rust-lang/rfcs/pull/560) (this has high
  overlap with the library subteam)
* [Sound generic drop](https://github.com/rust-lang/rfcs/pull/769)

### Library team

Oversees both `std` and, ultimately, other crates in the `rust-lang` github
organization. The focus up to this point has been the standard library, but we
will want "official" libraries that aren't quite `std` territory but are still
vital for Rust. (The precise plan here, as well as the long-term plan for `std`,
is one of the first important areas of debate for the subteam.) Also includes
API conventions.

Some example RFCs that fall into this area:

* [Collections reform](https://github.com/rust-lang/rfcs/pull/235)
* [IO reform](https://github.com/rust-lang/rfcs/pull/517/)
* [Debug improvements](https://github.com/rust-lang/rfcs/pull/640)
* [Simplifying std::hash](https://github.com/rust-lang/rfcs/pull/823)
* [Conventions for ownership variants](https://github.com/rust-lang/rfcs/pull/199)

### Compiler team

Focuses on compiler internals, including implementation of language
features. This broad category includes work in codegen, factoring of compiler
data structures, type inference, borrowck, and so on.

There is a more limited set of example RFCs for this subteam, in part because we
haven't generally required RFCs for this kind of internals work, but here are two:

* [Non-zeroing dynamic drops](https://github.com/rust-lang/rfcs/pull/320) (this
  has high overlap with language design)
* [Incremental compilation](https://github.com/rust-lang/rfcs/pull/594)

### Tooling and infrastructure team

Even more broad is the "tooling" subteam, which at inception is planned to
encompass every "official" (rust-lang managed) non-`rustc` tool:

* rustdoc
* rustfmt
* Cargo
* crates.io
* CI infrastructure
* Debugging tools
* Profiling tools
* Editor/IDE integration
* Refactoring tools

It's not presently clear exactly what tools will end up under this umbrella, nor
which should be prioritized.

### Moderation team

Finally, the moderation team is responsible for dealing with CoC violations.

One key difference from the other subteams is that the moderation team does not
have a leader. Its members are chosen directly by the core team, and should be
community members who have demonstrated the highest standard of discourse and
maturity. To limit conflicts of interest, **the moderation subteam should not
include any core team members**. However, the subteam is free to consult with
the core team as it deems appropriate.

The moderation team will have a public email address that can be used to raise
complaints about CoC violations (forwards to all active moderators).

#### Initial plan for moderation

What follows is an initial proposal for the mechanics of moderation. The
moderation subteam may choose to revise this proposal by drafting an RFC, which
will be approved by the core team.

Moderation begins whenever a moderator becomes aware of a CoC problem, either
through a complaint or by observing it directly. In general, the enforcement
steps are as follows:

> **These steps are adapted from text written by Manish Goregaokar, who helped
articulate them from experience as a Stack Exchange moderator.**

* Except for extreme cases (see below), try first to address the problem with a
  light public comment on thread, aimed to de-escalate the situation. These
  comments should strive for as much empathy as possible. Moderators should
  emphasize that dissenting opinions are valued, and strive to ensure that the
  technical points are heard even as they work to cool things down.

  When a discussion has just gotten a bit heated, the comment can just be a
  reminder to be respectful and that there is rarely a clear "right" answer. In
  cases that are more clearly over the line into personal attacks, it can
  directly call out a problematic comment.

* If the problem persists on thread, or if a particular person repeatedly comes
  close to or steps over the line of a CoC violation, moderators then email the
  offender privately. The message should include relevant portions of the CoC
  together with the offending comments. Again, the goal is to de-escalate, and
  the email should be written in a dispassionate and empathetic way. However,
  the message should also make clear that continued violations may result in a
  ban.

* If problems still persist, the moderators can ban the offender. Banning should
  occur for progressively longer periods, for example starting at 1 day, then 1
  week, then permanent. The moderation subteam will determine the precise
  guidelines here.

In general, moderators can and should unilaterally take the first step, but
steps beyond that (particularly banning) should be done via consensus with the
other moderators. Permanent bans require core team approval.

Some situations call for more immediate, drastic measures: deeply inappropriate
comments, harassment, or comments that make people feel unsafe. (See the
[code of conduct](http://www.rust-lang.org/conduct.html) for some more details
about this kind of comment). In these cases, an individual moderator is free to
take immediate, unilateral steps including redacting or removing comments, or
instituting a short-term ban until the subteam can convene to deal with the
situation.

The moderation team is responsible for interpreting the CoC. Drastic measures
like bans should only be used in cases of clear, repeated violations.

Moderators themselves are held to a very high standard of behavior, and should
strive for professional and impersonal interactions when dealing with a CoC
violation. They should always push to *de-escalate*. And they should recuse
themselves from moderation in threads where they are actively participating in
the technical debate or otherwise have a conflict of interest. Moderators who
fail to keep up this standard, or who abuse the moderation process, may be
removed by the core team.

Subteam, and especially core team members are *also* held to a high standard of
behavior. Part of the reason to separate the moderation subteam is to ensure
that CoC violations by Rust's leadership be addressed through the same
independent body of moderators.

Moderation covers all rust-lang venues, which currently include github
repos, IRC channels (#rust, #rust-internals, #rustc, #rust-libs), and
the two discourse forums. (The subreddit already has its own
moderation structure, and isn't directly associated with the rust-lang
organization.)

# Drawbacks

One possibility is that decentralized decisions may lead to a lack of coherence
in the overall design of Rust. However, the existence of the core team -- and
the fact that subteam leaders will thus remain in close communication on
cross-cutting concerns in particular -- serves to greatly mitigate that risk.

As with any change to governance, there is risk that this RFC would harm
processes that are working well. In particular, bringing on a large number of
new people into official decision-making roles carries a risk of culture clash
or problems with consensus-building.

By setting up this change as a relatively slow build-out from the current core
team, some of this risk is mitigated: it's not a radical restructuring, but
rather a refinement of the current process. In particular, today core team
members routinely seek input directly from other community members who would be
likely subteam members; in some ways, this RFC just makes that process more
official.

For the moderation subteam, there is a significant shift toward strong
enforcement of the CoC, and with that a risk of *over*-application: the goal is
to make discourse safe and productive, not to introduce fear of violating the
CoC. The moderation guidelines, careful selection of moderators, and ability to
withdraw moderators mitigate this risk.

# Alternatives

There are numerous other forms of open-source governance out there, far more
than we can list or detail here. And in any case, this RFC is intended as an
expansion of Rust's existing governance to address a few scaling problems,
rather than a complete rethink.

[Mozilla's module system][module], was a partial inspiration for this RFC. The
proposal here can be seen as an evolution of the module system where the subteam
leaders (module owners) are integrated into an explicit core team, providing for
tighter intercommunication and a more unified sense of vision and purpose.
Alternatively, the proposal is an evolution of the current core team structure
to include subteams.

One seemingly minor, but actually important aspect is *naming*:

* The name "subteam" (from [jQuery][jq]) felt like a better fit than "module" both
to avoid confusion (having two different kinds of modules associated with
Mozilla seems problematic) and because it emphasizes the more unified nature of
this setup.

* The term "leader" was chosen to reflect that there is a vision for each subteam
(as part of the larger vision for Rust), which the leader is responsible for
moving the subteam toward. Notably, this is how "module owner" is actually
defined in Mozilla's module system:

  > A "module owner" is the person to whom leadership of a module's work has been
  > delegated.

* The term "team member" is just following standard parlance. It could be
replaced by something like "peer" (following the module system tradition), or
some other term that is less bland than "member". Ideally, the term would
highlight the significant stature of team membership: being part of the
decision-making group for a substantial area of the Rust project.

[module]: https://wiki.mozilla.org/Modules
[jq]: https://jquery.org/team/
[mom]: https://wiki.mozilla.org/Modules/Activities#Module_Ownership_System

# Unresolved questions

## Subteams

This RFC purposefully leaves several subteam-level questions open:

* What is the exact venue and cadence for subteam decision-making?
* Do subteams have dedicated IRC channels or other forums? (This RFC stipulates
  only dedicated discourse tags.)
* How large is each subteam?
* What are the policies for when RFCs are required, or when PRs may be reviewed
  directly?

These questions are left to be address by subteams after their formation, in
part because good answers will likely require some iterations to discover.

## Broader questions

There are many other questions that this RFC doesn't seek to address, and this
is largely intentional. For one, it avoids trying to set out too much structure
in advance, making it easier to iterate on the mechanics of subteams. In
addition, there is a danger of *too much* policy and process, especially given
that this RFC is aimed to improve the scalability of decision-making. It should
be clear that this RFC is not the last word on governance, and over time we will
probably want to grow more explicit policies in other areas -- but a
lightweight, iterative approach seems the best way to get there.
