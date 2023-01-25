// revisions: mirunsafeck thirunsafeck
// [thirunsafeck]compile-flags: -Z thir-unsafeck

#![feature(rustc_attrs)]
#![allow(unused,dead_code)]

fn mtch() {
    #[rustc_layout_scalar_valid_range_start(1)]
    struct NonZero<T>(T);

    let mut foo = unsafe { NonZero((1_i32,) as _) };
    match &mut foo {
        NonZero((a,)) => *a = 0_i32 as _,
        //~^ ERROR: mismatched type
    }
}

fn main() {}
