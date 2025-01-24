#![feature(intrinsics)]

#[rustc_intrinsic]
#[rustc_intrinsic_must_be_overridden]
fn size_of<T, U>() -> usize {
    //~^ ERROR E0094
    loop {}
}

fn main() {}
