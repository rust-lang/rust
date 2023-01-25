// revisions: mirunsafeck thirunsafeck
// [thirunsafeck]compile-flags: -Z thir-unsafeck

#![feature(rustc_attrs)]
#![allow(unused,dead_code)]

fn tuple_struct() {
    #[rustc_layout_scalar_valid_range_start(1)]
    struct NonZero<T>(T);

    let mut foo = unsafe { NonZero((1,) as _) };
    let a = &mut foo.0.0;
    //~^ ERROR: type annotations needed
    //~| ERROR: no field `0` on type `_ is 1..`
}

fn main() {}
