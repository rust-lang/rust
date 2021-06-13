// Tests that `transmute` can be called in simple cases on types using type parameters.

// run-pass

use std::mem::transmute;

#[repr(transparent)]
struct Wrapper<T>(T);

fn wrap<T>(unwrapped: T) -> Wrapper<T> {
    unsafe {
        transmute(unwrapped)
    }
}

fn unwrap<T>(wrapped: Wrapper<T>) -> T {
    unsafe {
        transmute(wrapped)
    }
}

fn main() {
    let unwrapped = 5_u64;
    let wrapped = wrap(unwrapped);
    assert_eq!(5_u64, wrapped.0);

    let unwrapped = unwrap(wrapped);
    assert_eq!(5_u64, unwrapped);
}
