#![feature(intrinsics, rustc_attrs)]

extern "rust-intrinsic" {
    #[rustc_safe_intrinsic]
    fn size_of<T, U>() -> usize; //~ ERROR E0094
}

fn main() {
}
