- Start Date: 2014-06-30
- RFC PR #: https://github.com/rust-lang/rfcs/pull/560
- Rust Issue #: https://github.com/rust-lang/rust/issues/22020

# Summary

Change the semantics of the built-in fixed-size integer types from
being defined as wrapping around on overflow to it being considered a
program error (but *not* undefined behavior in the C
sense). Implementations are *permitted* to check for overflow at any
time (statically or dynamically). Implementations are *required* to at
least check dynamically when `debug_assert!` assertions are
enabled. Add a `WrappingOps` trait to the standard library with
operations defined as wrapping on overflow for the limited number of
cases where this is the desired semantics, such as hash functions.

# Motivation

Numeric overflow prevents a difficult situation. On the one hand,
overflow (and [underflow]) is known to be a common source of error in
other languages. Rust, at least, does not have to worry about memory
safety violations, but it is still possible for overflow to lead to
bugs. Moreover, Rust's safety guarantees do not apply to `unsafe`
code, which carries the
[same risks as C code when it comes to overflow][phrack]. Unfortunately,
banning overflow outright is not feasible at this time. Detecting
overflow statically is not practical, and detecting it dynamically can
be costly. Therefore, we have to steer a middle ground.

[phrack]: http://phrack.org/issues/60/10.html#article
[underflow]: http://google-styleguide.googlecode.com/svn/trunk/cppguide.html#Integer_Types

The RFC has several major goals:

1. Ensure that code which intentionally uses wrapping semantics is
   clearly identified.
2. Help users to identify overflow problems and help those who wish to
   be careful about overflow to do so.
3. Ensure that users who wish to detect overflow can safely enable
   overflow checks and dynamic analysis, both on their code and on
   libraries they use, with a minimal risk of "false positives"
   (intentional overflows leading to a panic).
4. To the extent possible, leave room in the future to move towards
   universal overflow checking if it becomes feasible. This may require
   opt-in from end-users.

To that end the RFC proposes two mechanisms:

1. Optional, dynamic overflow checking. Ordinary arithmetic operations
   (e.g., `a+b`) would conditionally check for overflow. If an
   overflow occurs when checking is enabled, a thread panic will be
   signaled. Specific intrinsics and library support are provided to
   permit either explicit overflow checks or explicit wrapping.
2. Overflow checking would be, by default, tied to debug assertions
   (`debug_assert!`). It can be seen as analogous to a debug
   assertion: an important safety check that is too expensive to
   perform on all code.

We expect that additional and finer-grained mechanisms for enabling
overflows will be added in the future. One easy option is a
command-line switch to enable overflow checking universally or within
specific crates. Another option might be lexically scoped annotations
to enable overflow (or perhaps disable) checking in specific
blocks. Neither mechanism is detailed in this RFC at this time.

## Why tie overflow checking to debug assertions

The reasoning behind connecting overflow checking and debug assertion
is that it ensures that pervasive checking for overflow is performed
*at some point* in the development cycle, even if it does not take
place in shipping code for performance reasons. The goal of this is to
prevent "lock-in" where code has a de-facto reliance on wrapping
semantics, and thus incorrectly breaks when stricter checking is
enabled.

We would like to allow people to switch "pervasive" overflow checks on
by default, for example. However, if the default is not to check for
overflow, then it seems likely that a pervasive check like that could
not be used, because libraries are sure to come to rely on wrapping
semantics, even if accidentally.

By making the default for debugging code be checked overflow, we help
ensure that users will encounter overflow errors in practice, and thus
become aware that overflow in Rust is not the norm. It will also help
debug simple errors, like unsigned underflow leading to an infinite
loop.

# Detailed design

## Arithmetic operations with error conditions

There are various operations which can sometimes produce error
conditions (detailed below). Typically these error conditions
correspond to under/overflow but not exclusively. It is the
programmers responsibility to avoid these error conditions: any
failure to do so can be considered a bug, and hence can be flagged by
a static/dynamic analysis tools as an error. This is largerly a
semantic distinction, though.

The result of an error condition depends upon the state of overflow
checking, which can be either *enabled* or *default* (this RFC does
not describe a way to disable overflow checking completely). If
overflow checking is *enabled*, then an error condition always results
in a panic. For efficiency reasons, this panic may be delayed over
some number of pure operations, as described below.

If overflow checking is *default*, that means that erroneous
operations will produce a value as specified below. Note though that
code which encounters an error condition is still considered buggy.
In particular, Rust source code (in particular library code) cannot
rely on wrapping semantics, and should always be written with the
assumption that overflow checking *may* be enabled. This is because
overflow checking may be enabled by a downstream consumer of the
library.

In the future, we could add some way to explicitly *disable* overflow
checking in a scoped fashion. In that case, the result of each error
condition would simply be the same as the optional state when no panic
occurs, and this would requests for override checking specified
elsewhere. However, no mechanism for disabling overflow checks is
provided by this RFC: instead, it is recommended that authors use the
wrapped primitives.

The error conditions that can arise, and their defined results, are as
follows. The intention is that the defined results are the same as the
defined results today. The only change is that now a panic may result.

- The operations `+`, `-`, `*`, `/`, `%` can underflow and
  overflow.
- Shift operations (`<<`, `>>`) can shift a value of width `N` by more
  than `N` bits.

## Enabling overflow checking

Compilers should present a command-line option to enable overflow
checking universally. Additionally, when building in a default "debug"
configuration (i.e., whenever `debug_assert` would be enabled),
overflow checking should be enabled by default, unless the user
explicitly requests otherwise. The precise control of these settings
is not detailed in this RFC.

The goal of this rule is to ensure that, during debugging and normal
development, overflow detection is on, so that users can be alerted to
potential overflow (and, in particular, for code where overflow is
expected and normal, they will be immediately guided to use the
wrapping methods introduced below). However, because these checks will
be compiled out whenever an optimized build is produced, final code
wilil not pay a performance penalty.

In the future, we may add additional means to control when overflow is
checked, such as scoped attributes or a global, independent
compile-time switch.

## Delayed panics

If an error condition should occur and a thread panic should result,
the compiler is not required to signal the panic at the precise point
of overflow. It is free to coalesce checks from adjacent pure
operations. Panics may never be delayed across an unsafe block nor may
they be skipped entirely, however. The precise details of how panics
may be deferred -- and the definition of a pure operation -- can be
hammered out over time, but the intention here is that, at minimum,
overflow checks for adjacent numeric operations like `a+b-c` can be
coallesced into a single check. Another useful example might be that,
when summing a vector, the final overflow check could be deferred
until the summation is complete.

## Methods for explicit wrapping arithmetic

For those use cases where explicit wraparound on overflow is required,
such as hash functions, we must provide operations with such
semantics. Accomplish this by providing the following methods defined
in the inherent impls for the various integral types.

```rust
impl i32 { // and i8, i16, i64, isize, u8, u32, u64, usize
    fn wrapping_add(self, rhs: Self) -> Self;
    fn wrapping_sub(self, rhs: Self) -> Self;
    fn wrapping_mul(self, rhs: Self) -> Self;
    fn wrapping_div(self, rhs: Self) -> Self;
    fn wrapping_rem(self, rhs: Self) -> Self;

    fn wrapping_lshift(self, amount: u32) -> Self;
    fn wrapping_rshift(self, amount: u32) -> Self;
}
```

These are implemented to preserve the pre-existing, wrapping semantics
unconditionally.

### `Wrapping<T>` type for convenience

For convenience, the `std::num` module also provides a `Wrapping<T>`
newtype for which the operator overloads are implemented using the
`WrappingOps` trait:

    pub struct Wrapping<T>(pub T);

    impl<T: WrappingOps> Add<Wrapping<T>, Wrapping<T>> for Wrapping<T> {
        fn add(&self, other: &Wrapping<T>) -> Wrapping<T> {
            self.wrapping_add(*other)
        }
    }

    // Likewise for `Sub`, `Mul`, `Div`, and `Rem`

Note that this is only for potential convenience. The type-based approach has the
drawback that e.g. `Vec<int>` and `Vec<Wrapping<int>>` are incompatible types.

## Lint

In general it seems inadvisable to use operations with error
conditions (like a naked `+` or `-`) in unsafe code. It would be
better to use explicit `checked` or `wrapped` operations as
appropriate. The same holds for destructors, since unwinding in
destructors is inadvisable. Therefore, the RFC recommends a lint be
added against such operations, defaulting to warn, though the details
(such as the name of this lint) are not spelled out.

# Drawbacks

**Making choices is hard.** Having to think about whether wraparound
arithmetic is appropriate may cause an increased cognitive
burden. However, wraparound arithmetic is almost never the intended
behavior. Therefore, programmers should be able to keep using the
built-in integer types and to not think about this. Where wraparound
semantics are required, it is generally a specialized use case with
the implementor well aware of the requirement.

**Loss of additive commutativity and benign overflows.** In some
cases, overflow behavior can be benign. For example, given an
expression like `a+b-c`, intermediate overflows are not harmful so
long as the final result is within the range of the integral type.  To
take advantage of this property, code would have to be written to use
the wrapping constructs, such as `a.wrapping_add(b).wrapping_sub(c)`.
However, this drawback is counterbalanced by the large number of
arithmetic expressions which do not have the same behavior when
overflow occurs. A common example is `(max+min)/2`, which is a typical
ingredient for [binary searches and the like][BS] and can lead to very
surprising behavior. Moreover, the use of `wrapping_add` and
`wrapping_sub` to highlight the fact that the intermediate result may
overflow seems potentially useful to an end-reader.

[BS]: http://googleresearch.blogspot.com/2006/06/extra-extra-read-all-about-it-nearly.html

**Danger of triggering additional panics from within unsafe code.**
This proposal creates more possibility for panics to occur, at least
when checks are enabled.  As usual, a panic at an inopportune time can
lead to bugs if code is not exception safe. This is particularly
worrisome in unsafe code, where crucial safety guarantees can be
violated. However, this danger already exists, as there are numerous
ways to trigger a panic, and hence unsafe code must be written with
this in mind.  It seems like the best advice is for unsafe code to
eschew the plain `+` and `-` operators, and instead prefer explicit
checked or wrapping operations as appropriate (hence the proposed
lint). Furthermore, the danger of an unexpected panic occurring in
unsafe code must be weighed against the danger of a (silent) overflow,
which can also lead to unsafety.

**Divergence of debug and optimized code.** The proposal here causes
additional divergence of debug and optimized code, since optimized
code will not include overflow checking. It would therefore be
recommended that robust applications run tests both with and without
optimizations (and debug assertions). That said, this state of affairs
already exists. First, the use of `debug_assert!` causes
debug/optimized code to diverge, but also, optimizations are known to
cause non-trivial changes in behavior. For example, recursive (but
pure) functions may be optimized away entirely by LLVM. Therefore, it
always makes sense to run tests in both modes. This situation is not
unique to Rust; most major projects do something similar. Moreover, in
most languages, `debug_assert!` is in fact the only (or at least
predominant) kind of of assertion, and hence the need to run tests
both with and without assertions enabled is even stronger.

**Benchmarking.** Someone may conduct a benchmark of Rust with
overflow checks turned on, post it to the Internet, and mislead the
audience into thinking that Rust is a slow language. The choice of
defaults minimizes this risk, however, since doing an optimized build
in cargo (which ought to be a prerequisite for any benchmark) also
disables debug assertions (or ought to).

**Impact of overflow checking on optimization.** In addition to the
direct overhead of checking for overflow, there is some additional
overhead when checks are enabled because compilers may have to forego
other optimizations or code motion that might have been legal. This
concern seems minimal since, in optimized builds, overflow checking
will not be enabled. Certainly if we ever decided to change the
default for overflow checking to *enabled* in optimized builds, we
would want to measure carefully and likely include some means of
disabling checks in particularly hot paths.

# Alternatives and possible future directions

## Do nothing for now

Defer any action until later, as advocated by:

 * [Patrick Walton on June 22][PW22]

Reasons this was not pursued: The proposed changes are relatively well-contained.
Doing this after 1.0 would require either breaking existing programs which rely
on wraparound semantics, or introducing an entirely new set of integer types and
porting all code to use those types, whereas doing it now lets us avoid
needlessly proliferating types. Given the paucity of circumstances where
wraparound semantics is appropriate, having it be the default is defensible only
if better options aren't available.

## Scoped attributes to control runtime checking

The [original RFC][GH] proposed a system of scoped attributes for
enabling/disabling overflow checking. Nothing in the current RFC
precludes us from going in this direction in the future. Rather, this
RFC is attempting to answer the question (left unanswered in the
original RFC) of what the behavior ought to be when no attribute is in
scope.

The proposal for scoped attributes in the original RFC was as follows.
Introduce an `overflow_checks` attribute which can be used to turn
runtime overflow checks on or off in a given
scope. `#[overflow_checks(on)]` turns them on,
`#[overflow_checks(off)]` turns them off. The attribute can be applied
to a whole `crate`, a `mod`ule, an `fn`, or (as per [RFC 40][40]) a
given block or a single expression. When applied to a block, this is
analogous to the `checked { }` blocks of C#. As with lint attributes,
an `overflow_checks` attribute on an inner scope or item will override
the effects of any `overflow_checks` attributes on outer scopes or
items. Overflow checks can, in fact, be thought of as a kind of
run-time lint. Where overflow checks are in effect, overflow with the
basic arithmetic operations and casts on the built-in fixed-size
integer types will invoke task failure. Where they are not, the checks
are omitted, and the result of the operations is left unspecified (but
will most likely wrap).

Significantly, turning `overflow_checks` on or off should only produce an
observable difference in the behavior of the program, beyond the time it takes
to execute, if the program has an overflow bug.

It should also be emphasized that `overflow_checks(off)` only disables *runtime*
overflow checks. Compile-time analysis can and should still be performed where
possible. Perhaps the name could be chosen to make this more obvious, such as
`runtime_overflow_checks`, but that starts to get overly verbose.

Illustration of use:

    // checks are on for this crate
    #![overflow_checks(on)]

    // but they are off for this module
    #[overflow_checks(off)]
    mod some_stuff {

        // but they are on for this function
        #[overflow_checks(on)]
        fn do_thing() {
            ...

            // but they are off for this block
            #[overflow_checks(off)] {
                ...
                // but they are on for this expression
                let n = #[overflow_checks(on)] (a * b + c);
                ...
            }

            ...
        }

        ...
    }

    ...

[40]: https://github.com/rust-lang/rfcs/blob/master/active/0040-more-attributes.md

## Checks off means wrapping on

If we adopted a model of overflow checks, one could use an explicit
request to turn overflow checks *off* as a signal that wrapping is
desirted. This would allow us to do without the `WrappingOps` trait
and to avoid having unspecified results. See:

 * [Daniel Micay on June 24][DM24_2]

Reasons this was not pursued: The official semantics of a type should not change
based on the context. It should be possible to make the choice between turning
checks `on` or `off` solely based on performance considerations. It should be
possible to distinguish cases where checking was too expensive from where
wraparound was desired. (Wraparound is not usually desired.)

## Different operators

Have the usual arithmetic operators check for overflow, and introduce a new set
of operators with wraparound semantics, as done by Swift. Alternately, do the
reverse: make the normal operators wrap around, and introduce new ones which
check.

Reasons this was not pursued: New, strange operators would pose an entrance
barrier to the language. The use cases for wraparound semantics are not common
enough to warrant having a separate set of symbolic operators.

## Different types

Have separate sets of fixed-size integer types which wrap around on overflow and
which are checked for overflow (e.g. `u8`, `u8c`, `i8`, `i8c`, ...).

Reasons this was not pursued: Programmers might be confused by having to choose
among so many types. Using different types would introduce compatibility hazards
to APIs. `Vec<u8>` and `Vec<u8c>` are incompatible. Wrapping arithmetic is not
common enough to warrant a whole separate set of types.

## Just use `Checked*`

Just use the existing `Checked` traits and a `Checked<T>` type after the same
fashion as the `Wrapping<T>` in this proposal.

Reasons this was not pursued: Wrong defaults. Doesn't enable distinguishing
"checking is slow" from "wrapping is desired" from "it was the default".

## Runtime-closed range types

[As proposed by Bill Myers.][BM-RFC]

Reasons this was not pursued: My brain melted. :(

## Making `as` be checked

The RFC originally specified that using `as` to convert between types
would cause checked semantics. However, we now use `as` as a primitive
type operator. This decision was discussed on the
[discuss message board][as].

The key points in favor of reverting `as` to its original semantics
were:

1. `as` is already a fairly low-level operator that can be used (for
   example) to convert between `*mut T` and `*mut U`.
2. `as` is the only way to convert types in constants, and hence it is
   important that it covers all possibilities that constants might
   need (eventually, [const fn][911] or other approaches may change
   this, but those are not going to be stable for 1.0).
3. The [type ascription RFC][803] set the precedent that `as` is used
   for "dangerous" coercions that require care.
4. Eventually, checked numeric conversions (and perhaps most or all
   uses of `as`) can be ergonomically added as methods.  The precise
   form of this will be resolved in the future. [const fn][911] can
   then allow these to be used in constant expressions.

[as]: http://internals.rust-lang.org/t/on-casts-and-checked-overflow/1710/
[803]: https://github.com/rust-lang/rfcs/pull/803
[911]: https://github.com/rust-lang/rfcs/pull/911

# Unresolved questions

The C semantics of wrapping operations in some cases are undefined:

- `INT_MIN / -1`, `INT_MIN % -1`
- Shifts by an excessive number of bits

This RFC takes no position on the correct semantics of these
operations, simply preserving the existing semantics. However, it may
be worth trying to define the wrapping semantics of these operations
in a portable way, even if that implies some runtime cost. Since these
are all error conditions, this is an orthogonal topic to the matter of
overflow.

# Future work

 * Look into adopting imprecise exceptions and a similar design to Ada's, and to
   what is explored in the research on AIR (As Infinitely Ranged) semantics, to
   improve the performance of checked arithmetic. See also:

     * [Cameron Zwarich on June 22][CZ22]
     * [John Regehr on June 23][JR23_2]

 * Make it easier to use integer types of unbounded size, i.e. actual
   mathematical integers and naturals.

[BM-RFC]: https://github.com/bill-myers/rfcs/blob/no-integer-overflow/active/0000-no-integer-overflow.md
[PW22]: https://mail.mozilla.org/pipermail/rust-dev/2014-June/010494.html
[DM24_2]: https://mail.mozilla.org/pipermail/rust-dev/2014-June/010590.html
[CZ22]: https://mail.mozilla.org/pipermail/rust-dev/2014-June/010483.html
[JR23_2]: https://mail.mozilla.org/pipermail/rust-dev/2014-June/010527.html

# Updates since being accepted

Since it was accepted, the RFC has been updated as follows:

1. The wrapping methods were moved to be inherent, since we gained the
   capability for libstd to declare inherent methods on primitive
   integral types.
2. `as` was changed to restore the behavior before the RFC (that is,
   it truncates, as a C cast would).
   

# Acknowledgements and further reading

This RFC was [initially written by Gábor Lehel][GH] and was since
edited by Nicholas Matsakis into its current form. Although the text
has changed significantly, the spirit of the original is preserved (at
least in our opinion). The primary changes from the original are:

1. Define the results of errors in some cases rather than using undefined values.
2. Move discussion of scoped attributes to the "future directions" section.
3. Define defaults for when overflow checking is enabled.

Many aspects of this proposal and many of the ideas within it were
influenced and inspired by
[a discussion on the rust-dev mailing list][GL18]. The author is
grateful to everyone who provided input, and would like to highlight
the following messages in particular as providing motivation for the
proposal.

On the limited use cases for wrapping arithmetic:

 * [Jerry Morrison on June 20][JM20]

On the value of distinguishing where overflow is valid from where it is not:

 * [Gregory Maxwell on June 18][GM18]
 * [Gregory Maxwell on June 24][GM24]
 * [Robert O'Callahan on June 24][ROC24]
 * [Jerry Morrison on June 24][JM24]

The idea of scoped attributes:

 * [Daniel Micay on June 23][DM23]

On the drawbacks of a type-based approach:

 * [Daniel Micay on June 24][DM24]

In general:

 * [John Regehr on June 23][JR23]
 * [Lars Bergstrom on June 24][LB24]

Further credit is due to the commenters in the [GitHub discussion thread][GH].

[GL18]: https://mail.mozilla.org/pipermail/rust-dev/2014-June/010363.html
[GM18]: https://mail.mozilla.org/pipermail/rust-dev/2014-June/010371.html
[JM20]: https://mail.mozilla.org/pipermail/rust-dev/2014-June/010410.html
[DM23]: https://mail.mozilla.org/pipermail/rust-dev/2014-June/010566.html
[JR23]: https://mail.mozilla.org/pipermail/rust-dev/2014-June/010558.html
[GM24]: https://mail.mozilla.org/pipermail/rust-dev/2014-June/010580.html
[ROC24]: https://mail.mozilla.org/pipermail/rust-dev/2014-June/010602.html
[DM24]: https://mail.mozilla.org/pipermail/rust-dev/2014-June/010598.html
[JM24]: https://mail.mozilla.org/pipermail/rust-dev/2014-June/010596.html
[LB24]: https://mail.mozilla.org/pipermail/rust-dev/2014-June/010579.html
[GH]: https://github.com/rust-lang/rfcs/pull/146
