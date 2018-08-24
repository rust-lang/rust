#![feature(intrinsics)]
extern "rust-intrinsic" {
    fn size_of<T, U>() -> usize; //~ ERROR E0094
}

fn main() {
}
