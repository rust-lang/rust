- Feature Name: expect_intrinsic
- Start Date: 2015-05-20
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Provide an intrinsic function for hinting the likelyhood of branches being taken.

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

Implement an `expect` intrinsic with the signature: `fn(bool, bool) -> bool`. The first argument is
the condition being tested, the second argument is the expected result. The return value is the
same as the first argument, meaning that `if foo == bar { .. }` can be simply replaced with
`if expect(foo == bar, false) { .. }`.

The expected value is required to be a constant value.

# Drawbacks

The second argument is required to be a constant value, which can't be easily expressed.

# Alternatives

Provide a pair of intrinsics `likely` and `unlikely`, these are the same as `expect` just with
`true` and `false` substituted in for the expected value, respectively.

# Unresolved questions

None.