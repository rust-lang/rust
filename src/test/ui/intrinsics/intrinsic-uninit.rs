// run-pass
// pretty-expanded FIXME #23616

#![feature(intrinsics)]

mod rusti {
    extern "rust-intrinsic" {
        pub fn uninit<T>() -> T;
    }
}
pub fn main() {
    let _a : isize = unsafe {rusti::uninit()};
}
