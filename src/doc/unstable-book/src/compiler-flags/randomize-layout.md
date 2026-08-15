# `randomize-layout`

The tracking issue for this feature is: [#106764](https://github.com/rust-lang/rust/issues/106764).

------------------------

The `-Zrandomize-layout` flag changes the layout algorithm for `repr(Rust)` types defined in the current crate from its normal
optimization goals to pseudorandomly rearranging fields within the degrees of freedom provided by the largely unspecified
default representation. This also affects type sizes and padding.
Downstream instantiations of generic types defined in a crate with randomization enabled will also be randomized.

It can be used to find unsafe code that accidentally relies on unspecified behavior.

Randomization is not guaranteed to use a different permutation for each compilation session.
`-Zlayout-seed=<u64>` can be used to supply additional entropy.

Randomization only approximates the intended freedom of repr(Rust). Sometimes two distinct types may still consistently
result in the same layout due to limitations of the current implementation. Randomization may become
more aggressive over time as our coverage of the available degrees of freedoms improves.
Corollary: Randomization is not a safety oracle. Two struct layouts being observably the same under different layout seeds
on the current compiler version does not guarantee that future compiler versions won't give them distinct layouts.

Randomization may also become less aggressive in the future if additional guarantees get added to the default layout.
