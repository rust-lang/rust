#![feature(intrinsics)]
#![feature(rustc_attrs)]

extern "rust-intrinsic" {
    #[rustc_safe_intrinsic]
    fn size_of<T>(); //~ ERROR E0308
}

fn main() {
}
