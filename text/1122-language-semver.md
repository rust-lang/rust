- Feature Name: N/A
- Start Date: 2015-05-07
- RFC PR: [rust-lang/rfcs#1122](https://github.com/rust-lang/rfcs/pull/1122)
- Rust Issue: N/A

# Summary

This RFC has the goal of defining what sorts of breaking changes we
will permit for the Rust language itself, and giving guidelines for
how to go about making such changes.
  
# Motivation

With the release of 1.0, we need to establish clear policy on what
precisely constitutes a "minor" vs "major" change to the Rust language
itself (as opposed to libraries, which are covered by [RFC 1105]).
**This RFC proposes that minor releases may only contain breaking
changes that fix compiler bugs or other type-system
issues**. Primarily, this means soundness issues where "innocent" code
can cause undefined behavior (in the technical sense), but it also
covers cases like compiler bugs and tightening up the semantics of
"underspecified" parts of the language (more details below).

However, simply landing all breaking changes immediately could be very
disruptive to the ecosystem. Therefore, **the RFC also proposes
specific measures to mitigate the impact of breaking changes**, and
some criteria when those measures might be appropriate.

In rare cases, it may be deemed a good idea to make a breaking change
that is not a soundness problem or compiler bug, but rather correcting
a defect in design. Such cases should be rare. But if such a change is
deemed worthwhile, then the guidelines given here can still be used to
mitigate its impact.

# Detailed design

The detailed design is broken into two major sections: how to address
soundness changes, and how to address other, opt-in style changes. We
do not discuss non-breaking changes here, since obviously those are
safe.

### Soundness changes

When compiler or type-system bugs are encountered in the language
itself (as opposed to in a library), clearly they ought to be
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

In cases where the impact seems larger, any effort to ease the
transition is sure to be welcome. The following are suggestions for
possible steps we could take (not all of which will be applicable to
all scenarios):

1. Identify important crates (such as those with many dependants)
   and work with the crate author to correct the code as quickly as
   possible, ideally before the fix even lands.
2. Work hard to ensure that the error message identifies the problem
   clearly and suggests the appropriate solution.
   - If we develop a rustfix tool, in some cases we may be able to
     extend that tool to perform the fix automatically.
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

- How important is the change?
  - Soundness holes that can be easily exploited or which impact
    running code are obviously much more concerning than minor corner
    cases. There is somewhat in tension with the other factors: if
    there is, for example, a widely deployed vulnerability, fixing
    that vulnerability is important, but it will also cause a larger
    disruption.
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
    
#### What is a "compiler bug" or "soundness change"?

In the absence of a formal spec, it is hard to define precisely what
constitutes a "compiler bug" or "soundness change" (see also the
section below on underspecified parts of the language). The obvious
cases are soundness violations in a rather strict sense:

- Cases where the user is able to produce Undefined Behavior (UB)
  purely from safe code.
- Cases where the user is able to produce UB using standard library
  APIs or other unsafe code that "should work".
    
However, there are other kinds of type-system inconsistencies that
might be worth fixing, even if they cannot lead directly to UB.  Bugs
in the coherence system that permit uncontrolled overlap between impls
are one example. Another example might be inference failures that
cause code to compile which should not (because ambiguities
exist). Finally, there is a list below of areas of the language which
are generally considered underspecified.

We expect that there will be cases that fall on a grey line between
bug and expected behavior, and discussion will be needed to determine
where it falls. The recent conflict between `Rc` and scoped threads is
an example of such a discusison: it was clear that both APIs could not
be legal, but not clear which one was at fault. The results of these
discussions will feed into the Rust spec as it is developed.
    
#### Opting out

In some cases, it may be useful to permit users to opt out of new type
rules. The intention is that this "opt out" is used as a temporary
crutch to make it easy to get the code up and running. Typically this
opt out will thus be removed in a later release. But in some cases,
particularly those cases where the severity of the problem is
relatively small, it could be an option to leave the "opt out"
mechanism in place permanently. In either case, use of the "opt out"
API would trigger the deprecation lint.

Note that we should make every effort to ensure that crates which
employ this opt out can be used compatibly with crates that do not.

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
- The treatment of hygiene in macros is uneven (see [#22462],
  [#24278]). In some cases, changes here may be backwards compatible,
  or may be more appropriate only with explicit opt-in (or perhaps an
  alternate macro system altogether, such as [this proposal][macro]).
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

There are other kinds of changes that can be made in a minor version
that may break unsafe code but which are not considered breaking
changes, because the unsafe code is relying on things known to be
intentionally unspecified. One obvious example is the layout of data
structures, which is considered undefined unless they have a
`#[repr(C)]` attribute.

Although it is not directly covered by this RFC, it's worth noting in
passing that some of the CLI flags to the compiler may change in the
future as well. The `-Z` flags are of course explicitly unstable, but
some of the `-C`, rustdoc, and linker-specific flags are expected to
evolve over time (see e.g. [#24451]).

# Drawbacks

The primary drawback is that making breaking changes are disruptive,
even when done with the best of intentions. The alternatives list some
ways that we could avoid breaking changes altogether, and the
downsides of each.

## Notes on phasing

# Alternatives

**Rather than simply fixing soundness bugs, we could issue new major
releases, or use some sort of opt-in mechanism to fix them
conditionally.** This was initially considered as an option, but
eventually rejected for the following reasons:

- Opting in to type system changes would cause deep splits between
  minor versions; it would also create a high maintenance burden in
  the compiler, since both older and newer versions would have to be
  supported.
- It seems likely that all users of Rust will want to know that their
  code is sound and would not want to be working with unsafe
  constructs or bugs.
- We already have several mitigation measures, such as opt-out or
  temporary deprecation, that can be used to ease the transition
  around a soundness fix. Moreover, separating out new type rules so
  that they can be "opted into" can be very difficult and would
  complicate the compiler internally; it would also make it harder to
  reason about the type system as a whole.

# Unresolved questions

**What precisely constitutes "small" impact?** This RFC does not
attempt to define when the impact of a patch is "small" or "not
small". We will have to develop guidelines over time based on
precedent. One of the big unknowns is how indicative the breakage we
observe on `crates.io` will be of the total breakage that will occur:
it is certainly possible that all crates on `crates.io` work fine, but
the change still breaks a large body of code we do not have access to.

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

**Are there any other circumstances in which we might perform a
breaking change?** In particular, it may happen from time to time that
we wish to alter some detail of a stable component. If we believe that
this change will not affect anyone, such a change may be worth doing,
but we'll have to work out more precise guidelines. [RFC 1156] is an
example.

[RFC 1105]: https://github.com/rust-lang/rfcs/pull/1105
[RFC 320]: https://github.com/rust-lang/rfcs/pull/320
[#744]: https://github.com/rust-lang/rfcs/issues/744
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
[macro]: https://internals.rust-lang.org/t/pre-rfc-macro-improvements/2088
[#24451]: https://github.com/rust-lang/rust/pull/24451
[RFC 1156]: https://github.com/rust-lang/rfcs/pull/1156
