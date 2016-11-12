- Feature Name: N/A
- Start Date: 2016-04-21
- RFC PR: https://github.com/rust-lang/rfcs/pull/1607
- Rust Issue: N/A


# Summary
[summary]: #summary

This RFC proposes a process for deciding detailed guidelines for code
formatting, and default settings for Rustfmt. The outcome of the process should
be an approved formatting style defined by a style guide and enforced by
Rustfmt.

This RFC proposes creating a new repository under the [rust-lang](https://github.com/rust-lang)
organisation called fmt-rfcs. It will be operated in a similar manner to the
[RFCs repository](https://github.com/rust-lang/rfcs), but restricted to
formatting issues. A new [sub-team](https://github.com/rust-lang/rfcs/blob/master/text/1068-rust-governance.md#subteams)
will be created to deal with those RFCs. Both the team and repository are
expected to be temporary. Once the style guide is complete, the team can be
disbanded and the repository frozen.


# Motivation
[motivation]: #motivation

There is a need to decide on detailed guidelines for the format of Rust code. A
uniform, language-wide formatting style makes comprehending new code-bases
easier and forestalls bikeshedding arguments in teams of Rust users. The utility
of such guidelines has been proven by Go, amongst other languages.

The [Rustfmt](https://github.com/rust-lang-nursery/rustfmt) tool is
[reaching maturity](https://users.rust-lang.org/t/please-help-test-rustfmt/5386)
and currently enforces a somewhat arbitrary, lightly discussed style, with many
configurable options.

If Rustfmt is to become a widely accepted tool, there needs to be a process for
the Rust community to decide on the default style, and how configurable that
style should be.

These discussions should happen in the open and be highly visible. It is
important that the Rust community has significant input to the process. The RFC
repository would be an ideal place to have this discussion because it exists to
satisfy these goals, and is tried and tested. However, the discussion is likely
to be a high-bandwidth one (code style is a contentious and often subjective
topic, and syntactic RFCs tend to be the highest traffic ones). Therefore,
having the discussion on the RFCs repository could easily overwhelm it and make
it less useful for other important discussions.

There currently exists a [style guide](https://github.com/rust-lang/rust/tree/master/src/doc/style)
as part of the Rust documentation. This is far more wide-reaching than just
formatting style, but also not detailed enough to specify Rustfmt. This was
originally developed in its [own repository](https://github.com/rust-lang/rust-guidelines),
but is now part of the main Rust repository. That seems like a poor venue for
discussion of these guidelines due to visibility.


# Detailed design
[design]: #detailed-design

## Process

The process for style RFCs will mostly follow the [process for other RFCs](https://github.com/rust-lang/rfcs).
Anyone may submit an RFC. An overview of the process is:

* If there is no single, obvious style, then open a GitHub issue on the
  fmt-rfcs repo for initial discussion. This initial discussion should identify
  which Rustfmt options are required to enforce the guideline.
* Implement the style in rustfmt (behind an option if it is not the current
  default). In exceptional circumstances (such as where the implementation would
  require very deep changes to rustfmt), this step may be skipped.
* Write an RFC formalising the formatting convention and referencing the
  implementation, submit as a PR to fmt-rfcs. The RFC should include the default
  values for options to enforce the guideline and which non-default options
  should be kept.
* The RFC PR will be triaged by the style team and either assigned to a team
  member for [shepherding](https://github.com/rust-lang/rfcs#the-role-of-the-shepherd),
  or closed.
* When discussion has reached a fixed point, the RFC PR will be put into a final
  comment period (FCP).
* After FCP, the RFC will either be accepted and merged or closed.
* Implementation in Rustfmt can then be finished (including any changes due to
  discussion of the RFC), and defaults are set.


### Scope of the process

This process is specifically limited to formatting style guidelines which can be
enforced by Rustfmt with its current architecture. Guidelines that cannot be
enforced by Rustfmt without a large amount of work are out of scope, even if
they only pertain to formatting.

Note whether Rustfmt should be configurable at all, and if so how configurable
is a decision that should be dealt with using the formatting RFC process. That
will be a rather exceptional RFC.

### Size of RFCs

RFCs should be self-contained and coherent, whilst being as small as possible to
keep discussion focused. For example, an RFC on 'arithmetic and logic
expressions' is about the right size; 'expressions' would be too big, and
'addition' would be too small.


### When is a guideline ready for RFC?

The purpose of the style RFC process is to foster an open discussion about style
guidelines. Therefore, RFC PRs should be made early rather than late. It is
expected that there may be more discussion and changes to style RFCs than is
typical for Rust RFCs. However, at submission, RFC PRs should be completely
developed and explained to the level where they can be used as a specification.

A guideline should usually be implemented in Rustfmt **before** an RFC PR is
submitted. The RFC should be used to select an option to be the default
behaviour, rather than to identify a range of options. An RFC can propose a
combination of options (rather than a single one) as default behaviour. An RFC
may propose some reorganisation of options.

Usually a style should be widely used in the community before it is submitted as
an RFC. Where multiple styles are used, they should be covered as alternatives
in the RFC, rather than being submitted as multiple RFCs. In some cases, a style
may be proposed without wide use (we don't want to discourage innovation),
however, it should have been used in *some* real code, rather than just being
sketched out.


### Triage

RFC PRs are triaged by the style team. An RFC may be closed during triage (with
feedback for the author) if the style team think it is not specified in enough
detail, has too narrow or broad scope, or is not appropriate in some way (e.g.,
applies to more than just formatting). Otherwise, the PR will be assigned a
shepherd as for other RFCs.


### FCP

FCP will last for two weeks (assuming the team decide to meet every two weeks)
and will be announced in the style team sub-team report.


### Decision and post-decision process

The style team will make the ultimate decision on accepting or closing a style
RFC PR. Decisions should be by consensus. Most discussion should take place on
the PR comment thread, a decision should ideally be made when consensus is
reached on the thread. Any additional discussion amongst the style team will be
summarised on the thread.

If an RFC PR is accepted, it will be merged. An issue for implementation will be
filed in the appropriate place (usually the Rustfmt repository) referencing the
RFC. If the style guide needs to be updated, then an issue for that should be
filed on the Rust repository.

The author of an RFC is not required to implement the guideline. If you are
interested in working on the implementation for an 'active' RFC, but cannot
determine if someone else is already working on it, feel free to ask (e.g. by
leaving a comment on the associated issue).


## The fmt-rfcs repository

The form of the fmt-rfcs repository will follow the rfcs repository. Accepted
RFCs will live in a `text` directory, the `README.md` will include information
taken from this RFC, there will be an RFC template in the root of the
repository. Issues on the repository can be used for placeholders for future
RFCs and for preliminary discussion.

The RFC format will be illustrated by the RFC template. It will have the
following sections:

* summary
* details
* implementation
* rationale
* alternatives
* unresolved questions

The 'details' section should contain examples of both what should and shouldn't
be done, cover simple and complex cases, and the interaction with other style
guidelines.

The 'implementation' section should specify how options must be set to enforce
the guideline, and what further changes (including additional options) are
required. It should specify any renaming, reorganisation, or removal of options.

The 'rationale' section should motivate the choices behind the RFC. It should
reference existing code bases which use the proposed style. 'Alternatives'
should cover alternative possible guidelines, if appropriate.

Guidelines may include more than one acceptable rule, but should offer
guidance for when to use each rule (which should be formal enough to be used by
a tool).

For example: 

> A struct literal must be formatted either on a single line (with
spaces after the opening brace and before the closing brace, and with fields
separated by commas and spaces), or on multiple lines (with one field per line
and newlines after the opening brace and before the closing brace). The former
approach should be used for short struct literals, the latter for longer struct
literals. For tools, the first approach should be used when the width of the
fields (excluding commas and braces) is 16 characters. E.g.,

> ```rust
let x = Foo { a: 42, b: 34 };
let y = Foo {
    a: 42,
    b: 34,
    c: 1000
};
```

(Note this is just an example, not a proposed guideline).

The repository in embryonic form lives at [nrc/fmt-rfcs](https://github.com/nrc/fmt-rfcs).
It illustrates what [issues](https://github.com/nrc/fmt-rfcs/issues/1) and
[PRs](https://github.com/nrc/fmt-rfcs/pull/2) might look like, as well as
including the RFC template. Note that typically there should be more discussion
on an issue before submitting an RFC PR.

The repository should be updated as this RFC develops, and moved to the rust-lang
GitHub organisation if this RFC is accepted.


## The style team

The style [sub-team](https://github.com/rust-lang/rfcs/blob/master/text/1068-rust-governance.md#subteams)
will be responsible for handling style RFCs and making decisions related to
code style and formatting.

Per the [governance RFC](https://github.com/rust-lang/rfcs/blob/master/text/1068-rust-governance.md),
the core team would pick a leader who would then pick the rest of the team. I
propose that the team should include members representative of the following
areas:

* Rustfmt,
* the language, tools, and libraries sub-teams (since each has a stake in code style),
* large Rust projects.

Because activity such as this hasn't been done before in the RUst community, it
is hard to identify suitable candidates for the team ahead of time. The team
will probably start small and consist of core members of the Rust community. I
expect that once the process gets underway the team can be rapidly expanded with
community members who are active in the fmt-rfcs repository (i.e., submitting
and constructively commenting on RFCs).

There will be a dedicated irc channel for discussion on formatting issues:
`#rust-style`.


## Style guide

The [existing style guide](https://github.com/rust-lang/rust/tree/master/src/doc/style)
will be split into two guides: one dealing with API design and similar issues
which will be managed by the libs team, and one dealing with formatting issues
which will be managed by the style team. Note that the formatting part of the
guide may include guidelines which are not enforced by Rustfmt. Those are outside
the scope of the process defined in this RFC, but still belong in that part of
the style guide.

When RFCs are accepted the style guide may need to be updated. Towards the end
of the process, the style team should audit and edit the guide to ensure it is a
coherent document.


## Material goals

Hopefully, the style guideline process will have limited duration, one year
seems reasonable. After that time, style guidelines for new syntax could be
included with regular RFCs, or the fmt-rfcs repository could be maintained in a
less active fashion.

At the end of the process, the fmt-rfcs repository should be a fairly complete
guide for formatting Rust code, and useful as a specification for Rustfmt and
tools with similar goals, such as IDEs. In particular, there should be a
decision made on how configurable Rustfmt should be, and an agreed set of
default options. The formatting style guide in the Rust repository should be a
more human-friendly source of formatting guidelines, and should be in sync with
the fmt-rfcs repo.


# Drawbacks
[drawbacks]: #drawbacks

This RFC introduces more process and bureaucracy, and requires more meetings for
some core Rust contributors. Precious time and energy will need to be devoted to
discussions.


# Alternatives
[alternatives]: #alternatives

Benevolent dictator - a single person dictates style rules which will be
followed without question by the community. This seems to work for Go, I suspect
it will not work for Rust.

Parliamentary 'democracy' - the community 'elects' a style team (via the usual
RFC consensus process, rather than actual voting). The style team decides on
style issues without an open process. This would be more efficient, but doesn't
fit very well with the open ethos of the Rust community.

Use the RFCs repo, rather than a new repo. This would have the benefit that
style RFCs would get more visibility, and it is one less place to keep track of
for Rust community members. However, it risks overwhelming the RFC repo with
style debate.

Use issues on Rustfmt. I feel that the discussions would not have enough
visibility in this fashion, but perhaps that can be addressed by wide and
regular announcement.

Use a book format for the style repo, rather than a collection of RFCs. This
would make it easier to see how the 'final product' style guide would look.
However, I expect there will be many issues that are important to be aware of
while discussing an RFC, that are not important to include in a final guide.

Have an existing team handle the process, rather than create a new style team.
Saves on a little bureaucracy. Candidate teams would be language and tools.
However, the language team has very little free bandwidth, and the tools team is
probably not broad enough to effectively handle the style decisions.


# Unresolved questions
[unresolved]: #unresolved-questions
