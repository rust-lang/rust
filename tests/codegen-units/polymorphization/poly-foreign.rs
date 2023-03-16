// aux-build:poly-dep.rs
// compile-flags: --crate-type=lib -Zprint-mono-items=eager -Zpolymorphize=on

extern crate poly_dep;

pub static FN1: fn() = poly_dep::foo::<i32>;
pub static FN2: fn() = poly_dep::foo::<u32>;

//~ MONO_ITEM static FN1
//~ MONO_ITEM static FN2
//~ MONO_ITEM fn poly_dep::foo::<T>
