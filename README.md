# Rust RFCs
[Rust RFCs]: #rust-rfcs

(jump forward to: [Table of Contents], [Active RFC List])

Many changes, including bug fixes and documentation improvements can be
implemented and reviewed via the normal GitHub pull request workflow.

Some changes though are "substantial", and we ask that these be put
through a bit of a design process and produce a consensus among the Rust
community and the [core team].

The "RFC" (request for comments) process is intended to provide a
consistent and controlled path for new features to enter the language
and standard libraries, so that all stakeholders can be confident about
the direction the language is evolving in.

## Active RFC List
[Active RFC List]: #active-rfc-list

* [0008-new-intrinsics.md](text/0008-new-intrinsics.md)
* [0016-more-attributes.md](text/0016-more-attributes.md)
* [0019-opt-in-builtin-traits.md](text/0019-opt-in-builtin-traits.md)
* [0066-better-temporary-lifetimes.md](text/0066-better-temporary-lifetimes.md)
* [0090-lexical-syntax-simplification.md](text/0090-lexical-syntax-simplification.md)
* [0107-pattern-guards-with-bind-by-move.md](text/0107-pattern-guards-with-bind-by-move.md)
* [0132-ufcs.md](text/0132-ufcs.md)
* [0135-where.md](text/0135-where.md)
* [0141-lifetime-elision.md](text/0141-lifetime-elision.md)
* [0195-associated-items.md](text/0195-associated-items.md)
* [0213-defaulted-type-params.md](text/0213-defaulted-type-params.md)
* [0320-nonzeroing-dynamic-drop.md](text/0320-nonzeroing-dynamic-drop.md)
* [0339-statically-sized-literals.md](text/0339-statically-sized-literals.md)
* [0385-module-system-cleanup.md](text/0385-module-system-cleanup.md)
* [0401-coercions.md](text/0401-coercions.md)
* [0447-no-unused-impl-parameters.md](text/0447-no-unused-impl-parameters.md)
* [0495-array-pattern-changes.md](text/0495-array-pattern-changes.md)
* [0501-consistent_no_prelude_attributes.md](text/0501-consistent_no_prelude_attributes.md)
* [0509-collections-reform-part-2.md](text/0509-collections-reform-part-2.md)
* [0517-io-os-reform.md](text/0517-io-os-reform.md)
* [0560-integer-overflow.md](text/0560-integer-overflow.md)
* [0769-sound-generic-drop.md](text/0769-sound-generic-drop.md)
* [0803-type-ascription.md](text/0803-type-ascription.md)
* [0809-box-and-in-for-stdlib.md](text/0809-box-and-in-for-stdlib.md)
* [0909-move-thread-local-to-std-thread.md](text/0909-move-thread-local-to-std-thread.md)
* [0968-closure-return-type-syntax.md](text/0968-closure-return-type-syntax.md)
* [0979-align-splitn-with-other-languages.md](text/0979-align-splitn-with-other-languages.md)

## Table of Contents
[Table of Contents]: #table-of-contents
* [Opening](#rust-rfcs)
* [Active RFC List]
* [Table of Contents]
* [When you need to follow this process]
* [Before creating an RFC]
* [What the process is]
* [The role of the shepherd]
* [The RFC life-cycle]
* [Reviewing RFC's]
* [Implementing an RFC]
* [RFC Postponement]
* [Help this is all too informal!]

## When you need to follow this process
[When you need to follow this process]: #when-you-need-to-follow-this-process

You need to follow this process if you intend to make "substantial"
changes to Rust, Cargo, Crates.io, or the RFC process itself. What constitutes
a "substantial" change is evolving based on community norms, but may include
the following.

   - Any semantic or syntactic change to the language that is not a bugfix.
   - Removing language features, including those that are feature-gated.
   - Changes to the interface between the compiler and libraries,
including lang items and intrinsics.
   - Additions to `std`

Some changes do not require an RFC:

   - Rephrasing, reorganizing, refactoring, or otherwise "changing shape
does not change meaning".
   - Additions that strictly improve objective, numerical quality
criteria (warning removal, speedup, better platform coverage, more
parallelism, trap more errors, etc.)
   - Additions only likely to be _noticed by_ other developers-of-rust,
invisible to users-of-rust.

If you submit a pull request to implement a new feature without going
through the RFC process, it may be closed with a polite request to
submit an RFC first.

## Before creating an RFC
[Before creating an RFC]: #before-creating-an-rfc

A hastily-proposed RFC can hurt its chances of acceptance. Low quality
proposals, proposals for previously-rejected features, or those that
don't fit into the near-term roadmap, may be quickly rejected, which
can be demotivating for the unprepared contributor. Laying some
groundwork ahead of the RFC can make the process smoother.

Although there is no single way to prepare for submitting an RFC, it
is generally a good idea to pursue feedback from other project
developers beforehand, to ascertain that the RFC may be desirable:
having a consistent impact on the project requires concerted effort
toward consensus-building.

The most common preparations for writing and submitting an RFC include
talking the idea over on #rust-internals, filing and discusssing ideas
on the [RFC issue tracker][issues], and occasionally posting
'pre-RFCs' on [the developer discussion forum][discuss] for early
review.

As a rule of thumb, receiving encouraging feedback from long-standing
project developers, and particularly members of the [core team][core]
is a good indication that the RFC is worth pursuing.

[issues]: https://github.com/rust-lang/rfcs/issues
[discuss]: http://discuss.rust-lang.org/
[core]: https://github.com/rust-lang/rust/wiki/Note-core-team

## What the process is
[What the process is]: #what-the-process-is

In short, to get a major feature added to Rust, one must first get the
RFC merged into the RFC repo as a markdown file. At that point the RFC
is 'active' and may be implemented with the goal of eventual inclusion
into Rust.

* Fork the RFC repo http://github.com/rust-lang/rfcs
* Copy `0000-template.md` to `text/0000-my-feature.md` (where
'my-feature' is descriptive. don't assign an RFC number yet).
* Fill in the RFC. Put care into the details: RFCs that do not
present convincing motivation, demonstrate understanding of the
impact of the design, or are disingenuous about the drawbacks or
alternatives tend to be poorly-received.
* Submit a pull request. As a pull request the RFC will receive design
feedback from the larger community, and the author should be prepared
to revise it in response.
* During Rust triage, the pull request will either be closed (for RFCs
that clearly will not be accepted) or assigned a *shepherd*. The
shepherd is a trusted developer who is familiar with the process, who
will help to move the RFC forward, and ensure that the right people
see and review it.
* Build consensus and integrate feedback. RFCs that have broad support
are much more likely to make progress than those that don't receive
any comments. The shepherd assigned to your RFC should help you get
feedback from Rust developers as well.
* The shepherd may schedule meetings with the author and/or relevant
stakeholders to discuss the issues in greater detail, and in some
cases the topic may be discussed at the larger [weekly meeting]. In
either case a summary from the meeting will be posted back to the RFC
pull request.
* Once both proponents and opponents have clarified and defended
positions and the conversation has settled, the shepherd will take it
to the [core team] for a final decision.
* Eventually, someone from the [core team] will either accept the RFC
by merging the pull request, assigning the RFC a number (corresponding
to the pull request number), at which point the RFC is 'active', or
reject it by closing the pull request.

## The role of the shepherd
[The role of the shepherd]: the-role-of-the-shepherd

During triage, every RFC will either be closed or assigned a shepherd.
The role of the shepherd is to move the RFC through the process. This
starts with simply reading the RFC in detail and providing initial
feedback. The shepherd should also solicit feedback from people who
are likely to have strong opinions about the RFC. Finally, when this
feedback has been incorporated and the RFC seems to be in a steady
state, the shepherd will bring it to the meeting. In general, the idea
here is to "front-load" as much of the feedback as possible before the
point where we actually reach a decision.

## The RFC life-cycle
[The RFC life-cycle]: #the-rfc-life-cycle

Once an RFC becomes active then authors may implement it and submit
the feature as a pull request to the Rust repo. Being 'active' is not
a rubber stamp, and in particular still does not mean the feature will
ultimately be merged; it does mean that in principle all the major
stakeholders have agreed to the feature and are amenable to merging
it.

Furthermore, the fact that a given RFC has been accepted and is
'active' implies nothing about what priority is assigned to its
implementation, nor does it imply anything about whether a Rust
developer has been assigned the task of implementing the feature.
While it is not *necessary* that the author of the RFC also write the
implementation, it is by far the most effective way to see an RFC
through to completion: authors should not expect that other project
developers will take on responsibility for implementing their accepted
feature.

Modifications to active RFC's can be done in followup PR's.  We strive
to write each RFC in a manner that it will reflect the final design of
the feature; but the nature of the process means that we cannot expect
every merged RFC to actually reflect what the end result will be at
the time of the next major release; therefore we try to keep each RFC
document somewhat in sync with the language feature as planned,
tracking such changes via followup pull requests to the document.

An RFC that makes it through the entire process to implementation is
considered 'complete' and is moved to the 'complete' folder; an RFC
that fails after becoming active is 'inactive' and moves to the
'inactive' folder.

## Reviewing RFC's
[Reviewing RFC's]: #reviewing-rfcs

While the RFC PR is up, the shepherd may schedule meetings with the
author and/or relevant stakeholders to discuss the issues in greater
detail, and in some cases the topic may be discussed at the larger
[weekly meeting]. In either case a summary from the meeting will be
posted back to the RFC pull request.

The core team makes final decisions about RFCs after the benefits and
drawbacks are well understood. These decisions can be made at any
time, but the core team will regularly issue decisions on at least a
weekly basis. When a decision is made, the RFC PR will either be
merged or closed, in either case with a comment describing the
rationale for the decision. The comment should largely be a summary of
discussion already on the comment thread.

## Implementing an RFC
[Implementing an RFC]: #implementing-an-rfc

Some accepted RFC's represent vital features that need to be
implemented right away. Other accepted RFC's can represent features
that can wait until some arbitrary developer feels like doing the
work. Every accepted RFC has an associated issue tracking its
implementation in the Rust repository; thus that associated issue can
be assigned a priority via the [triage process] that the team uses for
all issues in the Rust repository.

The author of an RFC is not obligated to implement it. Of course, the
RFC author (like any other developer) is welcome to post an
implementation for review after the RFC has been accepted.

If you are interested in working on the implementation for an 'active'
RFC, but cannot determine if someone else is already working on it,
feel free to ask (e.g. by leaving a comment on the associated issue).

## RFC Postponement
[RFC Postponement]: #rfc-postponement

Some RFC pull requests are tagged with the 'postponed' label when they
are closed (as part of the rejection process). An RFC closed with
“postponed” is marked as such because we want neither to think about
evaluating the proposal nor about implementing the described feature
until after the next major release, and we believe that we can afford
to wait until then to do so.

Usually an RFC pull request marked as “postponed” has already passed
an informal first round of evaluation, namely the round of “do we
think we would ever possibly consider making this change, as outlined
in the RFC pull request, or some semi-obvious variation of it.”  (When
the answer to the latter question is “no”, then the appropriate
response is to close the RFC, not postpone it.)


### Help this is all too informal!
[Help this is all too informal!]: #help-this-is-all-too-informal

The process is intended to be as lightweight as reasonable for the
present circumstances. As usual, we are trying to let the process be
driven by consensus and community norms, not impose more structure than
necessary.

[core team]: https://github.com/mozilla/rust/wiki/Note-core-team
[triage process]: https://github.com/rust-lang/rust/wiki/Note-development-policy#milestone-and-priority-nomination-and-triage
[weekly meeting]: https://github.com/rust-lang/meeting-minutes
