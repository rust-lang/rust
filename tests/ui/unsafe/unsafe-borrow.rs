#![feature(rustc_attrs)]
#![allow(unused,dead_code)]

fn tuple_struct() {
    #[rustc_layout_scalar_valid_range_start(1)]
    struct NonZero<T>(T);

    let mut foo = unsafe { NonZero((1,)) };
    let a = &mut foo.0.0;
    //~^ ERROR: mutation of layout constrained field is unsafe
}

fn slice() {
    #[rustc_layout_scalar_valid_range_start(1)]
    struct NonZero<'a, T>(&'a mut [T]);

    let mut nums = [1, 2, 3, 4];
    let mut foo = unsafe { NonZero(&mut nums[..]) };
    let a = &mut foo.0[2];
    // ^ not unsafe because there is an implicit dereference here
}

fn array() {
    #[rustc_layout_scalar_valid_range_start(1)]
    struct NonZero<T>([T; 4]);

    let nums = [1, 2, 3, 4];
    let mut foo = unsafe { NonZero(nums) };
    let a = &mut foo.0[2];
    //~^ ERROR: mutation of layout constrained field is unsafe
}

fn block() {
    #[rustc_layout_scalar_valid_range_start(1)]
    struct NonZero<T>(T);

    let foo = unsafe { NonZero((1,)) };
    &mut { foo.0 }.0;
    // ^ not unsafe because the result of the block expression is a new place
}

fn mtch() {
    #[rustc_layout_scalar_valid_range_start(1)]
    struct NonZero<T>(T);

    let mut foo = unsafe { NonZero((1,)) };
    match &mut foo {
        NonZero((a,)) => *a = 0,
        //~^ ERROR: mutation of layout constrained field is unsafe
    }
}

fn main() {}
