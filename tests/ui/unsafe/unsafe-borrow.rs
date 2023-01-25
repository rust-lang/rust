// revisions: mirunsafeck thirunsafeck
// [thirunsafeck]compile-flags: -Z thir-unsafeck

#![feature(rustc_attrs)]
#![allow(unused, dead_code)]

fn slice() {
    #[rustc_layout_scalar_valid_range_start(1)]
    struct NonZero<'a, T>(&'a mut [T]);

    let mut nums = [1, 2, 3, 4];
    let mut foo = unsafe { NonZero(&mut nums[..] as _) };
    let a = &mut foo.0[2];
    //~^ ERROR: mutation of layout constrained field is unsafe
}

fn array() {
    #[rustc_layout_scalar_valid_range_start(1)]
    struct NonZero<T>([T; 4]);

    let nums = [1, 2, 3, 4];
    let mut foo = unsafe { NonZero(nums as _) };
    let a = &mut foo.0[2];
    //~^ ERROR: mutation of layout constrained field is unsafe
}

fn block() {
    #[rustc_layout_scalar_valid_range_start(1)]
    struct NonZero<T>(T);

    let foo = unsafe { NonZero((1,) as _) };
    &mut { foo.0 as (i32,) }.0;
    // ^ not unsafe because the result of the block expression is a new place
}

fn main() {}
