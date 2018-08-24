#![feature(intrinsics)]
extern "rust-intrinsic" {
    fn foo();
    //~^ ERROR E0093
}

fn main() {
}
