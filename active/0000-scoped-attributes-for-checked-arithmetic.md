- Start Date: 2014-06-30
- RFC PR #: (leave this empty)
- Rust Issue #: (leave this empty)


# Summary

Change the semantics of the built-in fixed-size integer types from being defined
as wrapping around on overflow to either returning an unspecified result or not
returning at all. Allow overflow checks to be turned on or off per-scope with an
attribute. Add a compiler option to force checking for testing and debugging
purposes. Add a `WrappingOps` trait to the standard library, with operations
defined as wrapping on overflow, for the limited number of cases where this is
the desired semantics, such as hash functions.


# Motivation

The semantics of the basic arithmetic operators on the built-in fixed-size
integer types are currently defined to wrap around on overflow. Wrapping around
on overflow is well-defined behavior, which means that it's better than C. Yet
we should avoid falling prey to the soft bigotry of low expectations.

In the large majority of cases, wrapping around on overflow is not an
appropriate semantics: programs will generally not work correctly with the
wrapped-around value. Much of the time, programs are merely optimistic that
overflow won't happen, but this often turns out to be mistaken. When it does
happen, unexpected behavior and potentially even security bugs can result. It
should be emphasized that wrapping around on overflow is *not* a memory safety
issue in the safe subset of Rust, which greatly mitigates, but does not
eliminate, the harm that overflow bugs can cause. However, in `unsafe` blocks,
and when interfacing with existing C libraries, it can be a memory safety issue,
and furthermore, not all security bugs are memory safety bugs. By
indiscriminately using wraparound-on-overflow semantics in every case, whether
or not it is appropriate, it becomes difficult to impossible for programmers,
compilers, and analysis tools to reliably determine where overflow is the
expected behavior, and where the possibility of it should be considered a
defect.

It is a fact that checked arithmetic poses an unacceptable performance burden in
many cases, especially in a performance-sensitive language like Rust. As such, a
perfect, one-size-fits-all solution is regrettably not possible. However, we can
make better compromises than we currently do.

While in many cases, the performance cost of checking for overflow is not
acceptable, in many other cases, it is. Developers should be able to make the
tradeoff themselves in a convenient and granular way.

The cases where wraparound on overflow is explicitly desired are comparatively
rare. The only use cases for wrapping arithmetic that are known to the author at
the time of writing are hashes, checksums, and emulation of processors with
wraparound arithmetic instructions. Therefore, it should be acceptable to not 
provide symbolic operators, but require using named methods in
these cases.


## Goals of this proposal

 * Clearly distinguish the circumstances where overflow is expected behavior
   from where it is not.

 * Provide programmers with the tools they need to flexibly make the tradeoff
   between higher performance and catching more mistakes.

 * Be minimally disruptive to the language as it is and have a small surface
   area.


## Non-goals of this proposal

 * Make checked arithmetic fast. Let me emphasize: *the actual performance of
   checked arithmetic is wholly irrelevant to the content of this proposal*. It
   will be relevant to the decisions programmers make when employing the tools
   this proposal proposes to supply them with, but it is not to the tools
   themselves.

 * Prepare for future processor architectures and/or compiler advances which may
   improve the performance of checked arithmetic. If there were mathematical
   proof that faster checked arithmetic is impossible, the proposal would be the
   same.


## Acknowledgements and further reading

Many aspects of this proposal and many of the ideas within it were influenced
and inspired by [a discussion on the rust-dev mailing list][GL18]. The author is
grateful to everyone who provided input, and would like to highlight the
following messages in particular as providing motivation for the proposal.

On the limited use cases for wrapping arithmetic:

 * [Jerry Morrison on June 20][JM20]

On the value of distinguishing where overflow is valid, and where it is not:

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


# Detailed design

## Semantics of overflow with the built-in types

Currently, the built-in arithmetic operators `+`, `-`, `*`, `/`, and `%` on the
built-in types `i8`..`i64`, `u8`..`u64`, `int`, and `uint` are defined as
wrapping around on overflow. Change this to define them, on overflow, as either
returning an unspecified result, or not returning at all (i.e. terminating
execution in some fashion, "returning bottom"), instead.

The implication is that overflow is considered to be an abnormal circumstance,
and the programmer expects it not to happen, resp. it is her goal to make sure
that it will not.

Notes:

 * In practice, the unspecified result will most likely be the wraparound
   result, but in theory, it's up to the implementation.

 * "Terminating execution in some fashion" will most likely mean failing the
   task, but the defined semantics of the types do not foreclose on other
   possibilities.

 * Most importantly: this is **not** undefined behavior in the C sense. Only the
   result of the operation is left unspecified, as opposed to the entire
   program's meaning, as in C. The programmer would not be allowed to rely on a
   specific, or any, result being returned on overflow, but the compiler would
   also not be allowed to assume that overflow won't happen.


## Scoped attributes to control checking

This depends on [RFC PR 16][16] being accepted.

Introduce an `overflow_checks` attribute which can be used to turn overflow
checks on or off in a given scope. `#[overflow_checks(on)]` turns them on,
`#[overflow_checks(off)]` turns them off. The attribute can be applied to a 
whole `crate`, a `mod`ule, an `fn`, or (as per [RFC PR 16][16]) a given block or
a single expression. When applied to a block, this is analogous to the 
`checked { }` blocks of C#. As with lint attributes, an `overflow_checks`
attribute on an inner scope or item will override the effects of any 
`overflow_checks` attributes on outer scopes or items. (Overflow checks can in 
fact be thought of as a kind of run-time lint.) Where overflow checks are in 
effect, overflow with the basic arithmetic operations on the built-in fixed-size
integer types invokes `fail!()`. Where they are not, the checks are omitted, and
the result of the operations is left unspecified (but will most likely wrap).

Illustration:

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

[16]: https://github.com/rust-lang/rfcs/pull/16

### The default

There is a significant decision to be made with respect to the default behavior
where neither `overflow_checks(on)` nor `off` has been explicitly specified. The
author does not presume to know the correct answer, and leaves this open to
debate. The following defaults are possible:

 1. The default is `on`. This means that the default is to catch more mistakes.

 2. The default is `off`. This means that the default is to be faster. (This
    happens to be the current "default".)

 3. There is no default, and a decision is forced. If the programmer neglects to
    explicitly specify a behavior, the compiler will bail out and ask her to
    specify one.

 4. Combination of (1) and (3): The default is `on`, but the compiler emits a
    warning when falling back to the default behavior.

 5. Combination of (2) and (3): The default is `off`, but the compiler emits a
    warning when falling back to the default behavior.


## A debugging switch to force checking

The programmer has the option to turn `overflow_checks(off)` due to performance
considerations. However, when testing or debugging the program, for instance
when tracking down a difficult bug, it may be desired to throw performance to
the wind and enable as many checks as possible. For this purpose, provide a
compiler option, e.g. `--force-overflow-checks`, which causes overflow checks to
be considered `on` even where an attribute has turned them `off`. This is
somewhat analogous to the behavior of our current `--ndebug` flag and
`debug_assert!` macros.


## `WrappingOps` trait for explicit wrapping arithmetic

For those use cases where explicit wraparound on overflow is required, such as
hash functions, we must provide operations with such semantics. Accomplish this
by providing the following trait and impls in the `prelude`:

    pub trait WrappingOps {
        fn wrapping_add(self, rhs: Self) -> Self;
        fn wrapping_sub(self, rhs: Self) -> Self;
        fn wrapping_mul(self, rhs: Self) -> Self;
        fn wrapping_div(self, rhs: Self) -> Self;
        fn wrapping_rem(self, rhs: Self) -> Self;
    }

    impl WrappingOps for int
    impl WrappingOps for uint
    impl WrappingOps for i8
    impl WrappingOps for u8
    impl WrappingOps for i16
    impl WrappingOps for u16
    impl WrappingOps for i32
    impl WrappingOps for u32
    impl WrappingOps for i64
    impl WrappingOps for u64

These are implemented to wrap around on overflow unconditionally.


### `Wrapping<T>` type for convenience

For convenience, also provide a `Wrapping<T>` newtype for which the operator
overloads are implemented using the `WrappingOps` trait:

    pub struct Wrapping<T>(pub T);

    impl<T: WrappingOps> Add<Wrapping<T>, Wrapping<T>> for Wrapping<T> {
        fn add(&self, other: &Wrapping<T>) -> Wrapping<T> {
            self.wrapping_add(*other)
        }
    }

    // Likewise for `Sub`, `Mul`, `Div`, and `Rem`

Note that this is only for potential convenience. The type-based approach has the
drawback that e.g. `Vec<int>` and `Vec<Wrapping<int>>` are incompatible types.
The recommendation is to not use `Vec<Wrapping<int>>`, but to use `Vec<int>` and
the `wrapping_`* methods directly, instead.


# Drawbacks

 * Required implementation work:

   * Implement [RFC PR 16][16].

   * Implement the `overflow_checks` attribute.

   * Port existing code which relies on wraparound semantics (primarily hash
     functions) to use the `wrapping_`* methods.

 * Code where `overflow_checks(off)` is in effect could end up accidentally
   relying on overflow. Given the relative scarcity of cases where overflow is a
   favorable circumstance, the risk of this happening seems minor.

 * Having to think about whether wraparound arithmetic is appropriate may
   cause an increased cognitive burden. However, wraparound arithmetic is 
   almost never appropriate. Therefore, programmers should be able to keep using
   the built-in integer types and to not think about it. Where wraparound 
   semantics are required, it is generally a specialized use case with the
   implementor well aware of the requirement.

 * The built-in types become "special": the ability to control overflow checks
   using scoped attributes doesn't extend to user-defined types. If you make a
   `struct MyNum(int)` and `impl Add for MyNum` using the native `+` operation
   on `int`s, whether overflow checks happen for `MyNum` is determined by 
   whether `overflow_checks` is `on` or `off` where `impl Add for MyNum` is 
   declared, not whether they are `on` or `off` where the overloaded operators 
   are used. 

   The author considers this to be a serious shortcoming. *However*, it has the
   saving grace of being no worse than the status quo, i.e. the change is still
   a Pareto-improvement. Under the status quo, neither the built-in types nor 
   user-defined types can have overflow checks controlled by scoped attributes.
   Under this proposal, the situation is improved with built-in types gaining 
   this capability. In light of this, making further improvements, namely 
   extending the capability to user-defined types, can be left to future work.

 * Someone may conduct a benchmark of Rust with overflow checks turned on, post
   it to the Internet, and mislead the audience into thinking that Rust is a 
   slow language.


# Alternatives

## Do nothing for now

Defer any action until later, as suggested by:

 * [Patrick Walton on June 22][PW22]

Reasons this was not pursued: The proposed changes are relatively well-contained.
Doing this after 1.0 would require either breaking existing programs which rely
on wraparound semantics, or introducing an entirely new set of integer types and
porting all code to use those types, whereas doing it now lets us avoid
needlessly proliferating types. Given the paucity of circumstances where
wraparound semantics is appropriate, having it be the default is defensible only
if better options aren't available.

## Checks off means wrapping on

Where overflow checks are turned off, instead of defining overflow as returning
an unspecified result, define it to wrap around. This would allow us to do
without the `WrappingOps` trait and to avoid having unspecified results. See:

 * [Daniel Micay on June 24][DM24_2]

Reasons this was not pursued: Having the declared semantics of a type change
based on context is weird. It should be possible to make the choice between
turning checks `on` or `off` solely based on performance considerations. It
should be possible to distinguish cases where checking was too expensive and
where wraparound was desired. Wraparound is not usually desired.

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


# Unresolved questions

"What should the default be where neither `overflow_checks(on)` nor `off` has
been explicitly specified?", as discussed in the main text.


# Future work

 * Extend the ability to make use of local `overflow_checks(on|off)` attributes
   to user-defined types, as discussed under Drawbacks. (The author has some
   preliminary ideas, however, they are preliminary.)

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
