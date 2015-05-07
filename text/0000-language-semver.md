- Feature Name: N/A
- Start Date: 2015-05-07
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

This RFC has two main goals:

- define what precisely constitutes a breaking change for the Rust language itself;
- define a language versioning mechanism that extends the sorts of
  changes we can make without causing compilation failures (for
  example, adding new keywords).
  
# Motivation

With the release of 1.0, we need to establish clear policy on what
precisely constitutes a "minor" vs "major" change to the Rust language
itself (as opposed to libraries, which are covered by [RFC 1105]).
**This RFC proposes limiting breaking changes to changes with
soundness implications**: this includes both bug fixes in the compiler
itself, as well as changes to the type system or RFCs that are
necessary to close flaws uncovered later.

However, simply landing all breaking changes immediately could be very
disruptive to the ecosystem. Therefore, **the RFC also proposes
specific measures to mitigate the impact of breaking changes**, and
some criteria when those measures might be appropriate.

Furthermore, there are other kinds of changes that we may want to make
which feel like they *ought* to be possible, but which are in fact
breaking changes. The simplest example is adding a new keyword to the
language -- despite being a purely additive change, a new keyword can
of course conflict with existing identifiers. Therefore, **the RFC
proposes a simple annotation that allows crates to designate the
version of the language they were written for**. This effectively
permits some amount of breaking changes by making them "opt-in"
through the version attribute.

However, even though the version attribute can be used to make
breaking changes "opt-in" (and hence not really breaking), this is
still a tool to be used with great caution. Therefore, **the RFC also
proposes guidelines on when it is appropriate to include an "opt-in"
breaking change and when it is not**.

This RFC is focused specifically on the question of what kinds of
changes we can make within a single major version (as well as some
limited mechanisms that lay the groundwork for certain kinds of
anticipated changes). It intentionally does not address the question
of a release schedule for Rust 2.0, nor does it propose any new
features itself. These topics are complex enough to be worth
considering in separate RFCs.

# Detailed design

The detailed design is broken into two major section: how to address
soundness changes, and how to address other, opt-in style changes. We
do not discuss non-breaking changes here, since obviously those are
safe.

### Soundness changes

When compiler bugs or soundness problems are encountered in the
language itself (as opposed to in a library), clearly they ought to be
fixed. However, it is important to fix them in such a way as to
minimize the impact on the ecosystem.

The first step then is to evaluate the impact of the fix on the crates
found in the `crates.io` website (using e.g. the crater tool). If
impact is found to be "small" (which this RFC does not attempt to
precisely define), then the fix can simply be landed. As today, the
commit message of any breaking change should include the term
`[breaking-change]` along with a description of how to resolve the
problem, which helps those people who are affected to migrate their
code. A description of the problem should also appear in the relevant
subteam report.

In cases where the impact seems larger, the following steps can be
taken to ease the transition:

1. Identify important crates (such as those with many dependencies)
   and work with the crate author to correct the code as quickly as
   possible, ideally before the fix even lands.
2. Work hard to ensure that the error message identifies the problem
   clearly and suggests the appropriate solution.
3. Provide an annotation that allows for a scoped "opt out" of the
   newer rules, as described below. While the change is still
   breaking, this at least makes it easy for crates to update and get
   back to compiling status quickly.
4. Begin with a deprecation or other warning before issuing a hard
   error. In extreme cases, it might be nice to begin by issuing a
   deprecation warning for the unsound behavior, and only make the
   behavior a hard error after the deprecation has had time to
   circulate. This gives people more time to update their crates.
   However, this option may frequently not be available, because the
   source of a compilation error is often hard to pin down with
    precision.
   
Some of the factors that should be taken into consideration when
deciding whether and how to minimize the impact of a fix:

- How many crates on `crates.io` are affected?
  - This is a general proxy for the overall impact (since of course
    there will always be private crates that are not part of
    crates.io).
- Were particularly vital or widely used crates affected?
  - This could indicate that the impact will be wider than the raw
    number would suggest.
- Does the change silently change the result of running the program,
  or simply cause additional compilation failures?
  - The latter, while frustrating, are easier to diagnose.
- What changes are needed to get code compiling again? Are those
  changes obvious from the error message?
  - The more cryptic the error, the more frustrating it is when
    compilation fails.
    
#### Opting out

In some cases, it may be useful to permit users to opt out of new type
rules. The intention is that this "opt out" is used as a temporary
crutch to make it easy to get the code up and running. Depending on
the severity of the soundness fix, the "opt out" may be permanently
available, or it could be removed in a later release. In either case,
use of the "opt out" API would trigger the deprecation lint.

#### Changes that alter dynamic semantics versus typing rules

In some cases, fixing a bug may not cause crates to stop compiling,
but rather will cause them to silently start doing something different
than they were doing before. In cases like these, the same principle
of using mitigation measures to lessen the impact (and ease the
transition) applies, but the precise strategy to be used will have to
be worked out on a more case-by-case basis. This is particularly
relevant to the underspecified areas of the language described in the
next section.

Our approach to handling [dynamic drop][RFC 320] is a good
example. Because we expect that moving to the complete non-zeroing
dynamic drop semantics will break code, we've made an intermediate
change that
[altered the compiler to fill with use a non-zero value](https://github.com/rust-lang/rust/pull/23535),
which helps to expose code that was implicitly relying on the current
behavior (much of which has since been restructured in a more
future-proof way).

#### Underspecified language semantics

There are a number of areas where the precise language semantics are
currently somewhat underspecified. Over time, we expect to be fully
defining the semantics of all of these areas. This may cause some
existing code -- and in particular existing unsafe code -- to break or
become invalid. Changes of this nature should be treated as soundness
changes, meaning that we should attempt to mitigate the impact and
ease the transition wherever possible.

Known areas where change is expected include the following:

- Destructors semantics:
  - We plan to stop zeroing data and instead use marker flags on the stack,
    as specified in [RFC 320]. This may affect destructors that rely on ovewriting
    memory or using the `unsafe_no_drop_flag` attribute.
  - Currently, panicing in a destructor can cause unintentional memory
    leaks and other poor behavior (see [#14875], [#16135]). We are
    likely to make panic in a destructor simply abort, but the precise
    mechanism is not yet decided.
  - Order of dtor execution within a data structure is somewhat
    inconsistent (see [#744]).
- The legal aliasing rules between unsafe pointers is not fully settled (see [#19733]).
- The interplay of assoc types and lifetimes is not fully settled and can lead
  to unsoundness in some cases (see [#23442]).
- The trait selection algorithm is expected to be improved and made more complete over time.
  It is possible that this will affect existing code.
- [Overflow semantics][RFC 560]: in particular, we may have missed some cases.
- Memory allocation in unsafe code is currently unstable. We expect to
  be defining safe interfaces as part of the work on supporting
  tracing garbage collectors (see [#415]).
- The treatment of hygiene in macros is uneven (see [#22462], [#24278]). In some cases,
  changes here may be backwards compatible, or may be more appropriate only with explicit opt-in
  (or perhaps an alternate macro system altogether).
- The layout of data structures is expected to change over time unless they are annotated
  with a `#[repr(C)]` attribute.
- Lints will evolve over time (both the lints that are enabled and the
  precise cases that lints catch). We expect to introduce a
  [means to limit the effect of these changes on dependencies][#1029].
- Stack overflow is currently detected via a segmented stack check
  prologue and results in an abort. We expect to experiment with a
  system based on guard pages in the future.
- We currently abort the process on OOM conditions (exceeding the heap space, overflowing
  the stack). We may attempt to panic in such cases instead if possible.
- Some details of type inference may change. For example, we expect to
  implement the fallback mechanism described in [RFC 213], and we may
  wish to make minor changes to accommodate overloaded integer
  literals. In some cases, type inferences changes may be better
  handled via explicit opt-in.

(Although it is not directly covered by this RFC, it's worth noting in
passing that some of the CLI flags to the compiler may change in the
future as well. The `-Z` flags are of course explicitly unstable, but
some of the `-C`, rustdoc, and linker-specific flags are expected to
evolve over time.)

### Opt-in changes

For breaking changes that are not related to soundness or language
semantics, but are still deemed desirable, an opt-in strategy can be
used instead. This section describes an attribute for opting in to
newer language updates, and gives guidelines on what kinds of changes
should or should not be introduced in this fashion.

We use the term *"opt-in changes"* to refer to changes that would be
breaking changes, but are not because of the opt-in mechanism.

#### Rust version attribute

The specific proposal is an attribute `#![rust_version="X.Y"]` that
can be attached to the crate; the version `X.Y` in this attribute is
called the crate's "declared version". Every build of the Rust
compiler will also have a version number built into it reflecting the
current release.

When a `#[rust_version="X.Y"]` attribute is encountered, the compiler
will endeavor to produce the semantics of Rust "as it was" during
version `X.Y`. RFCs that propose opt-in changes should discuss how the
older behavior can be supported in the compiler, but this is expected
to be straightforward: if supporting older behavior is hard to do, it
may indicate that the opt-in change is too complex and should not be
accepted.

If the crate declares a version `X.Y` that is *newer* than the
compiler itself, the compiler should simply issue a warning and
proceed as if the crate had declared the compiler's version (i.e., the
newer version the compiler knows about).

Note that if the changes introducing by the Rust version `X.Y` affect
parsing, implementing these semantics may require some limited amount
of feedback between the parser and the tokenizer, or else a limited
"pre-parse" to scan the set of crate attributes and extract the
version. For example, if version `X.Y` adds new keywords, the
tokenizer will likely need to be configured appropriately with the
proper set of keywords. For this reason, it may make sense to require
that the `#![rust_version]` attribute appear *first* on the crate.

#### When opt-in changes are appropriate

Opt-in changes allow us to greatly expand the scope of the kinds of
additions we can make without breaking existing code, but they are not
applicable in all situations. A good rule of thumb is that an opt-in
change is only appropriate if the exact effect of the older code can
be easily recreated in the newer system with only surface changes to
the syntax.

Another view is that opt-in changes are appropriate if those changes
do not affect the "abstract AST" of your Rust program. In other words,
existing Rust syntax is just a serialization of a more idealized view
of the syntax, in which there are no conflicts between keywords and
identifiers, syntactic sugar is expanded, and so forth. Opt-in changes
might affect the translation into this abstract AST, but should not
affect the semantics of the AST itself at a deeper level. This concept
of an idealized AST is analagous to the "elaborated syntax" described
in [RFC 1105], except that it is at a conceptual level.

So, for example, the conflict between new keywords and existing
identifiers can (generally) be trivially worked around by renaming
identifiers, though the question of public identifiers is an
interesting one (contextual keywords may suffice, or else perhaps some
kind of escaping syntax -- we defer this question here for a later
RFC).

In the previous section on breaking changes, we identified various
criteria that can be used to decide how to approach a breaking change
(i.e., how far to go in attempting to mitigate the fallout). For the
most part, those same criteria also apply when deciding whether to
accept an "opt-in" change:

- How many crates on `crates.io` would break if they "opted-in" to the
  change, and would opting in require extensive changes?
- Does the change silently change the result of running the program,
  or simply cause additional compilation failures?
  - Opt-in changes that silently change the result of running the
    program are particularly unlikely to be accepted.
- What changes are needed to get code compiling again? Are those
  changes obvious from the error message?

# Drawbacks

**Allowing unsafe code to continue compiling -- even with warnings --
raises the probability that people experiences crashes and other
undesirable effects while using Rust.** However, in practice, most
unsafety hazards are more theoretical than practical: consider the
problem with the `thread::scoped` API. To actually create a data-race,
one had to place the guard into an `Rc` cycle, which would be quite
unusual. Therefore, a compromise path that warns about bad content but
provides an option for gradual migration seems preferable.

**Deprecation implies that a maintenance burden.** For library APIs,
this is relatively simple, but for type-system changes it can be quite
onerous. We may want to consider a policy for dropping older,
deprecated type-system rules after some time, as discussed in the
section on *unresolved questions*.

## Notes on phasing

# Alternatives

**Rather than supporting opt-in changes, one might consider simply
issuing a new major release for every such change.** Put simply,
though, issuing a new major release just because we want to have a new
keyword feels like overkill. This seems like to have two potential
negative effects. It may simply cause us to not make some of the
changes we would make otherwise, or work harder to fit them within the
existing syntactic constraints. It may also serve to dilute the
meaning of issuing a new major version, since even additive changes
that do not affect existing code in any meaningful way would result in
a major release. One would then be tempted to have some *additional*
numbering scheme, PR blitz, or other means to notify people when a new
major version is coming that indicates deeper changes.

**Rather than simply fixing soundness bugs, we could use the opt-in
mechanism to fix them conditionally.** This was initially considered
as an option, but eventually rejected for the following reasons:

- This would effectively cause a deeper split between minor versions;
  currently, opt-in is limited to "surface changes" only, but allowing
  opt-in to affect the type system feels like it would be creating two
  distinct languages.
- It seems likely that all users of Rust will want to know that their code
  is sound and would not want to be working with unsafe constructs or bugs.
- Users may choose not to opt-in to newer versions because they do not
  need the new features introduced there or because they wish to
  preserve compatibility with older compilers. It would be sad for
  them to lose the benefits of bug fixes as well.
- We already have several mitigation measures, such as opt-out or
  temporary deprecation, that can be used to ease the transition
  around a soundness fix. Moreover, separating out new type rules so
  that they can be "opted into" can be very difficult and would
  complicate the compiler internally; it would also make it harder to
  reason about the type system as a whole.

**Rather than using a version number to opt-in to minor changes, one
might consider using the existing feature mechanism.** For example,
one could write `#![feature(foo)]` to opt in to the feature "foo" and
its associated keywords and type rules, rather than
`#![rust_version="1.2.3"]`. While using minimum version numbers is
more opaque than named features, they do offer several advantages:

1. Using a version number alone makes it easy to think about what
   version of Rust you are using as a conceptual unit, rather than
   choosing features "a la carte".
2. Using named features, the list of features that must be attached to
   Rust code will grow indefinitely, presuming your crate wants to
   stay up to date.
3. Using a version attribute preserves a mental separation between
   "experimental work" (feature gates) and stable, new features.
4. Named features present a combinatoric testing problem, where we
   should (in principle) test for all possible combinations of
   features.
   
# Unresolved questions

**Can (and should) we give a more precise definition for compiler bugs
and soundness problems?** The current text is vague on what precisely
constitutes a compiler bug and soundness change. It may be worth
defining more precisely, though likely this would be best done as part
of writing up a more thorough (and authoritative) Rust reference
manual.

**Should we add a mechanism for "escaping" keywords?"** We may need a
mechanism for escaping keywords in the future. Imagine you have a
public function named `foo`, and we add a keyword `foo`. Now, if you
opt in to the newer version of Rust, your function declaration is
illegal: but if you rename the function `foo`, you are making a
breaking change for your clients, which you may not wish to do. If we
had an escaping mechanism, you would probably still want to deprecate
`foo` in favor of a new function `bar` (since typing `foo` would be
awkward), but it could still exist.

**Should we add a mechanism for skipping over new syntax?** The
current `#[cfg]` mechanism is applied *after* parsing. This implies
that if we add new syntax, crates which employ that new syntax will
not be parsable by older compilers, even if the modules that depend on
that new syntax are disabled via `#[cfg]` directives. It may be useful
to add some mechanism for informing the parser that it should skip
over sections of the input (presumably based on token trees). One
approach to this might just be modifying the existing `#[cfg]`
directives so that they are applied during parsing rather than as a
post-pass.

**What precisely constitutes "small" impact?** This RFC does not
attempt to define when the impact of a patch is "small" or "not
small". We will have to develop guidelines over time based on
precedent. One of the big unknowns is how indicative the breakage we
observe on `crates.io` will be of the total breakage that will occur:
it is certainly possible that all crates on `crates.io` work fine, but
the change still breaks a large body of code we do not have access to.

**Should deprecation due to unsoundness have a special lint?** We may
not want to use the same deprecation lint for unsoundness that we use
for everything else.

**What attribute should we use to "opt out" of soundness changes?**
The section on breaking changes indicated that it may sometimes be
appropriate to includ an "opt out" that people can use to temporarily
revert to older, unsound type rules, but did not specify precisely
what that opt-out should look like. Ideally, we would identify a
specific attribute in advance that will be used for such purposes.  In
the past, we have simply created ad-hoc attributes (e.g.,
`#[old_orphan_check]`), but because custom attributes are forbidden by
stable Rust, this has the unfortunate side-effect of meaning that code
which opts out of the newer rules cannot be compiled on older
compilers (even though it's using the older type system rules). If we
introduce an attribute in advance we will not have this problem.

[RFC 1105]: https://github.com/rust-lang/rfcs/pull/1105
[RFC 320]: https://github.com/rust-lang/rfcs/pull/320
[#774]: https://github.com/rust-lang/rfcs/issues/744
[#14875]: https://github.com/rust-lang/rust/issues/14875
[#16135]: https://github.com/rust-lang/rust/issues/16135
[#19733]: https://github.com/rust-lang/rust/issues/19733
[#23442]: https://github.com/rust-lang/rust/issues/23442
[RFC 213]: https://github.com/rust-lang/rfcs/pull/213
[#415]: https://github.com/rust-lang/rfcs/issues/415
[#22462]: https://github.com/rust-lang/rust/issues/22462#issuecomment-81756673
[#24278]: https://github.com/rust-lang/rust/issues/24278
[#1029]: https://github.com/rust-lang/rfcs/issues/1029
[RFC 560]: https://github.com/rust-lang/rfcs/pull/560
