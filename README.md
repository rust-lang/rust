# Rust RFCs

Many changes, including bug fixes and documentation improvements can be 
implemented and reviewed via the normal GitHub pull request workflow.

Some changes though are "substantial", and we ask that these be put 
through a bit of a design process and produce a consensus among the Rust 
community and the [core team].

The "RFC" (request for comments) process is intended to provide a
consistent and controlled path for new features to enter the language 
and standard libraries, so that all stakeholders can be confident about 
the direction the language is evolving in.

## When you need to follow this process

You need to follow this process if you intend to make "substantial" 
changes to the Rust distribution. What constitutes a "substantial" 
change is evolving based on community norms, but may include the following.

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

## What the process is

In short, to get a major feature added to Rust, one must first get the 
RFC merged into the RFC repo as a markdown file. At that point the RFC 
is 'active' and may be implemented with the goal of eventual inclusion 
into Rust.

* Fork the RFC repo http://github.com/rust-lang/rfcs
* Copy `0000-template.md` to `active/0000-my-feature.md` (where 
'my-feature' is descriptive. don't assign an RFC number yet).
* Fill in the RFC
* Submit a pull request. The pull request is the time to get review of 
the design from the larger community.
* Build consensus and integrate feedback. RFCs that have broad support 
are much more likely to make progress than those that don't receive any 
comments.
* Eventually, somebody on the [core team] will either accept the RFC by 
merging the pull request and assigning the RFC a number, at which point 
the RFC is 'active', or reject it by closing the pull request.

## The RFC life-cycle

Once an RFC becomes active then authors may implement it and submit the 
feature as a pull request to the Rust repo. An 'active' is not a rubber 
stamp, and in particular still does not mean the feature will ultimately 
be merged; it does mean that in principle all the major stakeholders 
have agreed to the feature and are amenable to merging it.

Furthermore, the fact that a given RFC has been accepted and is
'active' implies nothing about what priority is assigned to its
implementation, nor does it imply anything about whether a Rust
developer has been assigned the task of implementing the feature.

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

## Implementing an RFC

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

## Reviewing RFC's

Each week the [core team] will attempt to review some set of open RFC
pull requests.  The choice of pull requests to review is largely
driven by an informal estimate of whether its associated comment
thread has reached a steady state (i.e. either died out, or not
showing any sign of providing feedback improvements to the RFC
itself). The list of RFC's up for review is posted a week ahead of
time via standard notification channels (currently the 'rust-dev'
mailing list as well as the http://discuss.rust-lang.org/ discourse
site).

We try to make sure that any RFC that we accept is accepted at the
Tuesday team meeting, with a formal record of discussion regarding
acceptance.  We do not accept RFC’s at the Thursday triage meeting.
We may reject RFC’s at either meeting; in other words, the only RFC
activity on Thursdays is closing the ones that have reached a steady
state and that the team agrees we will not be adopting.

At both meetings, we try to only consider RFC’s for which at least a
few participants have read the corresponding discussion thread and are
prepared to represent the viewpoints presented there. One participant
should act as a "champion" for the feature.  The "champion" need not
*personally* desire the feature; they just need to act to represent
its virtues and the community’s desire for it.

## RFC Postponement

Some RFC pull requests are tagged with the 'postponed' label when they
are closed (as part of the rejection process).  An RFC closed with
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

The process is intended to be as lightweight as reasonable for the 
present circumstances. As usual, we are trying to let the process be 
driven by consensus and community norms, not impose more structure than 
necessary.

[core team]: https://github.com/mozilla/rust/wiki/Note-core-team
[triage process]: https://github.com/rust-lang/rust/wiki/Note-development-policy#milestone-and-priority-nomination-and-triage