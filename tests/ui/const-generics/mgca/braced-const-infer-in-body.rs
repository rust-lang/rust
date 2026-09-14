//! Regression test for: https://github.com/rust-lang/rust/issues/160798
#![crate_type = "lib"]
#![feature(min_generic_const_args)]
#![feature(macroless_generic_const_args)]

trait Trait<T> {}

impl Trait<i32> for i32 {}

struct S<const N: usize>;

fn main() {
    // Const-only infer args used for a type parameter are rejected.
    let _z: &[&dyn Trait<{ _ }>] = &[&0i32];
    //~^ ERROR: constant provided when a type was expected
    let _y: &dyn Trait<core::direct_const_arg!(_)> = &0i32;
    //~^ ERROR: constant provided when a type was expected

    let _a: S<{ _ }> = S::<3>;
    let _b: S<core::direct_const_arg!(_)> = S::<3>;
    let _c: S<{ core::direct_const_arg!(_) }> = S::<3>;
    let _d: S<_> = S::<3>;
}
