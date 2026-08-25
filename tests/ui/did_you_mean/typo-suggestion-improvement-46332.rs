// https://github.com/rust-lang/rust/issues/46332
// Original Levenshtein distance for both of this is 1. We improved accuracy with
// additional case insensitive comparison.

struct TyUint {}

struct TyInt {}

fn main() {
    TyUInt {};
    //~^ ERROR cannot find struct, variant or union type `TyUInt` in this scope
}
