#![feature(intrinsics)]

extern "rust-intrinsic" {
    fn size_of<T>(); //~ ERROR E0308
}

fn main() {
}
