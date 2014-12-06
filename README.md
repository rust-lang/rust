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

* [0002-rfc-process.md](text/0002-rfc-process.md)
* [0008-new-intrinsics.md](text/0008-new-intrinsics.md)
* [0016-more-attributes.md](text/0016-more-attributes.md)
* [0019-opt-in-builtin-traits.md](text/0019-opt-in-builtin-traits.md)
* [0048-traits.md](text/0048-traits.md)
* [0066-better-temporary-lifetimes.md](text/0066-better-temporary-lifetimes.md)
* [0090-lexical-syntax-simplification.md](text/0090-lexical-syntax-simplification.md)
* [0107-pattern-guards-with-bind-by-move.md](text/0107-pattern-guards-with-bind-by-move.md)
* [0114-closures.md](text/0114-closures.md)
* [0131-target-specification.md](text/0131-target-specification.md)
* [0132-ufcs.md](text/0132-ufcs.md)
* [0135-where.md](text/0135-where.md)
* [0141-lifetime-elision.md](text/0141-lifetime-elision.md)
* [0151-capture-by-value.md](text/0151-capture-by-value.md)
* [0195-associated-items.md](text/0195-associated-items.md)
* [0198-slice-notation.md](text/0198-slice-notation.md)
* [0199-ownership-variants.md](text/0199-ownership-variants.md)
* [0201-error-chaining.md](text/0201-error-chaining.md)
* [0212-restore-int-fallback.md](text/0212-restore-int-fallback.md)
* [0216-collection-views.md](text/0216-collection-views.md)
* [0230-remove-runtime.md](text/0230-remove-runtime.md)
* [0231-upvar-capture-inference.md](text/0231-upvar-capture-inference.md)
* [0235-collections-conventions.md](text/0235-collections-conventions.md)
* [0236-error-conventions.md](text/0236-error-conventions.md)
* [0240-unsafe-api-location.md](text/0240-unsafe-api-location.md)
* [0339-statically-sized-literals.md](text/0339-statically-sized-literals.md)
* [0344-conventions-galore.md](text/0344-conventions-galore.md)
* [0356-no-module-prefixes.md](text/0356-no-module-prefixes.md)
* [0369-num-reform.md](text/0369-num-reform.md)
* [0378-expr-macros.md](text/0378-expr-macros.md)
* [0385-module-system-cleanup.md](text/0385-module-system-cleanup.md)
* [0401-coercions.md](text/0401-coercions.md)
* [0430-finalizing-naming-conventions.md](text/0430-finalizing-naming-conventions.md)
* [0438-precedence-of-plus.md](text/0438-precedence-of-plus.md)
* [0439-cmp-ops-reform.md](text/0439-cmp-ops-reform.md)
* [0450-un-feature-gate-some-more-gates.md](text/0450-un-feature-gate-some-more-gates.md)
* [0459-disallow-shadowing.md](text/0459-disallow-shadowing.md)
* [0463-future-proof-literal-suffixes.md](text/0463-future-proof-literal-suffixes.md)
* [0490-dst-syntax.md](text/0490-dst-syntax.md)

## Table of Contents
[Table of Contents]: #table-of-contents
* [Opening](#rust-rfcs)
* [Active RFC List]
* [Table of Contents]
* [When you need to follow this process]
* [What the process is]
* [The role of the shepherd]
* [The RFC life-cycle]
* [Implementing an RFC]
* [Reviewing RFC's]
* [RFC Postponement]
* [Help this is all too informal!]

## When you need to follow this process
[When you need to follow this process]: #when-you-need-to-follow-this-process

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
[What the process is]: #what-the-process-is
In short, to get a major feature added to Rust, one must first get the
RFC merged into the RFC repo as a markdown file. At that point the RFC
is 'active' and may be implemented with the goal of eventual inclusion
into Rust.

* Fork the RFC repo http://github.com/rust-lang/rfcs
* Copy `0000-template.md` to `text/0000-my-feature.md` (where
'my-feature' is descriptive. don't assign an RFC number yet).
* Fill in the RFC
* Submit a pull request. The pull request is the time to get review of
the design from the larger community.
* During Rust triage, the pull request will either be closed or
assigned a shepherd. The shepherd will help to move the RFC forward,
* Build consensus and integrate feedback. RFCs that have broad support
are much more likely to make progress than those that don't receive
any comments. The shepherd assigned to your RFC should help you get
feedback from Rust developers as well.
* Eventually, somebody on the [core team] will either accept the RFC by
merging the pull request and assigning the RFC a number, at which point
the RFC is 'active', or reject it by closing the pull request.

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

## Reviewing RFC's
[Reviewing RFC's]: #reviewing-rfcs

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
should act as a "shepherd" for the feature.  The shepherd need not
*personally* desire the feature; they just need to act to represent
its virtues and the community’s desire for it.

## RFC Postponement
[RFC Postponement]: #rfc-postponement

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
[Help this is all too informal!]: #help-this-is-all-too-informal

The process is intended to be as lightweight as reasonable for the
present circumstances. As usual, we are trying to let the process be
driven by consensus and community norms, not impose more structure than
necessary.

[core team]: https://github.com/mozilla/rust/wiki/Note-core-team
[triage process]: https://github.com/rust-lang/rust/wiki/Note-development-policy#milestone-and-priority-nomination-and-triage
