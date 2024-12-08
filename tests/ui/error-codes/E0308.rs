#![feature(intrinsics)]
#![feature(rustc_attrs)]

#[rustc_intrinsic]
#[rustc_intrinsic_must_be_overridden]
fn size_of<T>() {
    //~^ ERROR E0308
    loop {}
}

fn main() {}
