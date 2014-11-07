- Start Date: 2014-09-03
- RFC PR: https://github.com/rust-lang/rfcs/pull/212
- Rust Issue: https://github.com/rust-lang/rust/issues/16968

# Summary

Restore the integer inference fallback that was removed. Integer
literals whose type is unconstrained will default to `i32`, unlike the
previous fallback to `int`.
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

The primary reason for this is that, in production code, the `i32`
fallback is very rarely used. In a sense, the same [measurements][m]
that were used to justify removing the `int` fallback also justify
keeping it. As the measurements showed, the vast, vast majority of
integer literals wind up with a constrained type, unless they are only
used to print out and do assertions with. Specifically, any integer
that is passed as a parameter, returned from a function, or stored in
a struct or array, must wind up with a specific type.

## Rationale for the choice of defaulting to `i32`

In contrast to the first revision of the RFC, the fallback type
suggested is `i32`. This is justified by a case analysis which showed
that there does not exist a compelling reason for having a signed
pointer-sized integer type as the default.

There are reasons *for* using `i32` instead: It's familiar to programmers
from the C programming language (where the default int type is 32-bit in
the major calling conventions), it's faster than 64-bit integers in
arithmetic today, and is superior in memory usage while still providing
a reasonable range of possible values.

To expand on the perfomance argument: `i32` obviously uses half of the
memory of `i64` meaning half the memory bandwidth used, half as much
cache consumption and twice as much vectorization – additionally
arithmetic (like multiplication and division) is faster on some of the
modern CPUs.

## Case analysis

This is an analysis of cases where `int` inference might be thought of
as useful:

**Indexing into an array with unconstrained integer literal:**

```
let array = [0u8, 1, 2, 3];
let index = 3;
array[index]
```

In this case, `index` is already automatically inferred to be a `uint`.

**Using a default integer for tests, tutorials, etc.:** Examples of this
include "The Guide", the Rust API docs and the Rust standard library
unit tests. This is better served by a smaller, faster and platform
independent type as default.

**Using an integer for an upper bound or for simply printing it:** This
is also served very well by `i32`.

**Counting of loop iterations:** This is a part where `int` is as badly
suited as `i32`, so at least the move to `i32` doesn't create new
hazards (note that the number of elements of a vector might not
necessarily fit into an `int`).

In addition to all the points above, having a platform-independent type
obviously results in less differences between the platforms in which the
programmer "doesn't care" about the integer type they are using.

## Future-proofing for overloaded literals

It is possible that, in the future, we will wish to allow vector and
strings literals to be overloaded so that they can be resolved to
user-defined types. In that case, for backwards compatibility, it will
be necessary for those literals to have some sort of fallback type.
(This is a relatively weak consideration.)

# Detailed design

Integral literals are currently type-checked by creating a special
class of type variable. These variables are subject to unification as
normal, but can only unify with integral types. This RFC proposes
that, at the end of type inference, when all constraints are known, we
will identify all integral type variables that have not yet been bound
to anything and bind them to `i32`. Similarly, floating point literals
will fallback to `f64`.

For those who wish to be very careful about which integral types they
employ, a new lint (`unconstrained_literal`) will be added which
defaults to `allow`. This lint is triggered whenever the type of an
integer or floating point literal is unconstrained.

# Downsides

Although there seems to be little motivation for `int` to be the
default, there might be use cases where `int` is a more correct fallback
than `i32`.

Additionally, it might seem weird to some that `i32` is a default, when
`int` looks like the default from other languages. The name of `int`
however is not in the scope of this RFC.


# Alternatives

- **No fallback.** Status quo.

- **Fallback to something else.** We could potentially fallback to
  `int` like the original RFC suggested or some other integral type
  rather than `i32`.

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

# History

2014-11-07: Changed the suggested fallback from `int` to `i32`, add
rationale.
  
[m]: https://gist.github.com/nikomatsakis/11179747
