# `re_rebalance_coherence`

The tracking issue for this feature is: [#55437]

[#55437]: https://github.com/rust-lang/rust/issues/55437

------------------------

The `re_rebalance_coherence` feature tweaks the rules regarding which trait
impls are allowed in crates.
The following rule is used:

Given `impl<P1..=Pn> Trait<T1..=Tn> for T0`, an impl is valid only if at
least one of the following is true:
- `Trait` is a local trait
- All of
  - At least one of the types `T0..=Tn` must be a local type. Let `Ti` be the
    first such type.
  - No uncovered type parameters `P1..=Pn` may appear in `T0..Ti` (excluding
    `Ti`)


See the [RFC](https://github.com/rust-lang/rfcs/blob/master/text/2451-re-rebalancing-coherence.md) for details.
