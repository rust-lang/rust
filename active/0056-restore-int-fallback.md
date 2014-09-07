- Start Date: 2014-09-03
- RFC PR: https://github.com/rust-lang/rfcs/pull/212
- Rust Issue: https://github.com/rust-lang/rust/issues/16968

# Summary

Restore the integer inference fallback that was removed. Integer
literals whose type is unconstrained will default to `int`, as before.
Floating point literals will default to `f64`.

# Motivation

## History lesson

Rust has had a long history with integer and floating-point
literals. Initial versions of Rust required *all* literals to be
explicitly annotated with a suffix (if no suffix is provided, then
`int` or `float` was used; note that the `float` type has since been
removed). This meant that, for example, if one wanted to count up all
the numbers in a list, one would write `0u` and `1u` so as to employ
unsigned integers:

    let mut count = 0u; // let `count` be an unsigned integer
    while cond() {
        ...
        count += 1u;    // `1u` must be used as well
    }

This was particularly troublesome with arrays of integer literals,
which could be quite hard to read:

    let byte_array = [0u8, 33u8, 50u8, ...];
    
It also meant that code which was very consciously using 32-bit or
64-bit numbers was hard to read.

Therefore, we introduced integer inference: unlabeled integer literals
are not given any particular integral type rather a fresh "integral
type variable" (floating point literals work in an analogous way). The
idea is that the vast majority of literals will eventually interact
with an actual typed variable at some point, and hence we can infer
what type they ought to have. For those cases where the type cannot be
automatically selected, we decided to fallback to our older behavior,
and have integer/float literals be typed as `int`/`float` (this is also what Haskell
does). Some time later, we did [various measurements][m] and found
that in real world code this fallback was rarely used. Therefore, we
decided that to remove the fallback.

## Experience with lack of fallback

Unfortunately, when doing the measurements that led us to decide to
remove the `int` fallback, we neglected to consider coding "in the
small" (specifically, we did not include tests in the
measurements). It turns out that when writing small programs, which
includes not only "hello world" sort of things but also tests, the
lack of integer inference fallback is quite annoying. This is
particularly troublesome since small program are often people's first
exposure to Rust. The problems most commonly occur when integers are
"consumed" by printing them out to the screen or by asserting
equality, both of which are very common in small programs and testing.

There are at least three common scenarios where fallback would be
beneficial:

**Accumulator loops.** Here a counter is initialized to `0` and then
incremented by `1`. Eventually it is printed or compared against
a known value.

```
let mut c = 0;
loop {
    ...;
    c += 1;
}
println!("{}", c); // Does not constrain type of `c`
assert_eq(c, 22);
```

**Calls to range with constant arguments.** Here a call to range like
`range(0, 10)` is used to execute something 10 times. It is important
that the actual counter is either unused or only used in a print out
or comparison against another literal:

```
for _ in range(0, 10) {
}
```

**Large constants.** In small tests it is convenient to make dummy
test data. This frequently takes the form of a vector or map of ints.

```
let mut m = HashMap::new();
m.insert(1, 2);
m.insert(3, 4);
assert_eq(m.find(&3).map(|&i| i).unwrap(), 4);
```

## Lack of bugs

To our knowledge, there has not been a single bug exposed by removing
the fallback to the `int` type. Moreover, such bugs seem to be
extremely unlikely.

The primary reason for this is that, in production code, the `int`
fallback is very rarely used. In a sense, the same [measurements][m]
that were used to justify removing the `int` fallback also justify
keeping it. As the measurements showed, the vast, vast majority of
integer literals wind up with a constrained type, unless they are only
used to print out and do assertions with. Specifically, any integer
that is passed as a parameter, returned from a function, or stored in
a struct or array, must wind up with a specific type.

Another secondary reason is that the lint which checks that literals
are suitable for their assigned type will catch cases where very large
literals were used that overflow the `int` type (for example,
`INT_MAX`+1). (Note that the overflow lint constraints `int` literals
to 32 bits for better portability.)

In almost all of common cases we described above, there exists *some*
large constant representing a bound. If this constant exceeds the
range of the chosen fallback type, then a `type_overflow` lint warning
would be triggered. For example, in the accumulator, if the
accumulated result `i` is compared using a call like `assert_eq(i,
22)`, then the constant `22` will be linted. Similarly, when invoking
range with unconstrained arguments, the arguments to range are linted.
And so on.

The only common case where the lint does not apply is when an
accumulator result is only being printed to the screen or otherwise
consumed by some generic function which never stores it to memory.
This is a very narrow case.

## Future-proofing for overloaded literals

It is possible that, in the future, we will wish to allow vector and
strings literals to be overloaded so that they can be resolved to
user-defined types. In that case, for backwards compatibility, it will
be necessary for those literals to have some sort of fallback type.
(This is a relatively weak consideration.)

# Detailed design

Integeral literals are currently type-checked by creating a special
class of type variable. These variables are subject to unification as
normal, but can only unify with integral types. This RFC proposes
that, at the end of type inference, when all constraints are known, we
will identify all integral type variables that have not yet been bound
to anything and bind them to `int`. Similarly, floating point literals
will fallback to `f64`.

For those who wish to be very careful about which integral types they
employ, a new lint (`unconstrained_literal`) will be added which
defaults to `allow`. This lint is triggered whenever the type of an
integer or floating point literal is unconstrained.

# Downsides

Although we give a detailed argument for why bugs are unlikely, it is
nonetheless possible that this choice will lead to bugs in some code,
since another choice (most likely `uint`) may have been more suitable.

Given that the size of `int` is platform dependent, it is possible
that a porting hazard is created. This is mitigated by the fact that
the `type_overflow` lint constraints `int` literals to 32 bits.

# Alternatives

- **No fallback.** Status quo.

- **Fallback to something else.** We could potentially fallback to
  `i32` or some other integral type rather than `int`.
  
- **Fallback in a more narrow range of cases.** We could attempt to
  identify integers that are "only printed" or "only compared". There
  is no concrete proposal in this direction and it seems to lead to an
  overly complicated design.
  
- **Default type parameters influencing inference.** There is a
  separate, follow-up proposal being prepared that uses default type
  parameters to influence inference. This would allow some examples,
  like `range(0, 10)` to work even without integral fallback, because
  the `range` function itself could specify a fallback type. However,
  this does not help with many other examples.
  
[m]: https://gist.github.com/nikomatsakis/11179747
