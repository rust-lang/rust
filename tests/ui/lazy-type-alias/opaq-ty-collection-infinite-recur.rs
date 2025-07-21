// The opaque type collector used to expand free alias types (in situ) without guarding against
// endlessly recursing aliases which lead to the compiler overflowing its stack in certain
// situations.
//
// In most situations we wouldn't even reach the collector when there's an overflow because we
// would've already bailed out early during the item's wfcheck due to the normalization failure.
//
// In the case below however, while collecting the opaque types defined by the AnonConst, we
// descend into its nested items (here: type alias `Recur`) to acquire their opaque types --
// meaning we get there before we wfcheck `Recur`.
//
// issue: <https://github.com/rust-lang/rust/issues/131994>
#![feature(lazy_type_alias)]
#![expect(incomplete_features)]

struct Hold([(); { type Recur = Recur; 0 }]); //~ ERROR overflow normalizing the type alias `Recur`

fn main() {}
