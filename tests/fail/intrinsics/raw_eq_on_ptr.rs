#![feature(intrinsics)]

extern "rust-intrinsic" {
    fn raw_eq<T>(a: &T, b: &T) -> bool;
}

fn main() {
    let x = &0;
    // FIXME: the error message is not great (should be UB rather than 'unsupported')
    unsafe { raw_eq(&x, &x) }; //~ERROR: unsupported operation
}
