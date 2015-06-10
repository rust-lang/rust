- Feature Name: expect_intrinsic
- Start Date: 2015-05-20
- RFC PR: [rust-lang/rfcs#1131](https://github.com/rust-lang/rfcs/pull/1131)
- Rust Issue: [rust-lang/rust#26179](https://github.com/rust-lang/rust/issues/26179)

# Summary

Provide a pair of intrinsic functions for hinting the likelyhood of branches being taken.

# Motivation

Branch prediction can have significant effects on the running time of some code. Especially tight
inner loops which may be run millions of times. While in general programmers aren't able to
effectively provide hints to the compiler, there are cases where the likelyhood of some branch
being taken can be known.

For example: in arbitrary-precision arithmetic, operations are often performed in a base that is
equal to `2^word_size`. The most basic division algorithm, "Schoolbook Division", has a step that
will be taken in `2/B` cases (where `B` is the base the numbers are in), given random input. On a
32-bit processor that is approximately one in two billion cases, for 64-bit it's one in 18
quintillion cases.

# Detailed design

Implement a pair of intrinsics `likely` and `unlikely`, both with signature `fn(bool) -> bool`
which hint at the probability of the passed value being true or false. Specifically, `likely` hints
to the compiler that the passed value is likely to be true, and `unlikely` hints that it is likely
to be false. Both functions simply return the value they are passed.

The primary reason for this design is that it reflects common usage of this general feature in many
C and C++ projects, most of which define simple `LIKELY` and `UNLIKELY` macros around the gcc
`__builtin_expect` intrinsic. It also provides the most flexibility, allowing branches on any
condition to be hinted at, even if the process that produced the branched-upon value is
complex. For why an equivalent to `__builtin_expect` is not being exposed, see the Alternatives
section.

There are no observable changes in behaviour from use of these intrinsics. It is valid to implement
these intrinsics simply as the identity function. Though it is expected that the intrinsics provide
information to the optimizer, that information is not guaranteed to change the decisions the
optimiser makes.

# Drawbacks

The intrinsics cannot be used to hint at arms in `match` expressions. However, given that hints
would need to be variants, a simple intrinsic would not be sufficient for those purposes.

# Alternatives

Expose an `expect` intrinsic. This is what gcc/clang does with `__builtin_expect`. However there is
a restriction that the second argument be a constant value, a requirement that is not easily
expressible in Rust code. The split into `likely` and `unlikely` intrinsics reflects the strategy
we have used for similar restrictions like the ordering constraint of the atomic intrinsics.

# Unresolved questions

None.
