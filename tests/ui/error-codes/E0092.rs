#![feature(intrinsics)]
extern "rust-intrinsic" {
    fn atomic_foo(); //~ ERROR E0092
}

fn main() {
}
