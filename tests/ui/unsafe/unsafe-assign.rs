// revisions: mirunsafeck thirunsafeck
// [thirunsafeck]compile-flags: -Z thir-unsafeck

#![feature(rustc_attrs)]
#![allow(unused,dead_code)]

fn nested_field() {
    #[rustc_layout_scalar_valid_range_start(1)]
    struct NonZero<T>(T);

    let mut foo = unsafe { NonZero((1_i32,) as _) };
    foo.0.0 = 0_i32;
    //~^ ERROR: type annotations needed
    //~| ERROR: no field `0` on type `_ is 1..`
}

fn block() {
    #[rustc_layout_scalar_valid_range_start(1)]
    struct NonZero<T>(T);

    let mut foo = unsafe { NonZero((1_i32,) as _) };
    { foo.0 as (_,) }.0 = 0_i32;
    // ^ not unsafe because the result of the block expression is a new place
}

fn main() {}
