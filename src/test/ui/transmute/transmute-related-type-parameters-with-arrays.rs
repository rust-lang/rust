// Tests that `transmute` can be called in simple cases on types using type parameters with arrays.

// run-pass

use std::mem::transmute;

#[repr(transparent)]
struct Wrapper<T>([T; 10]);

#[repr(transparent)]
struct OtherWrapper<T>([T; 10]);

#[repr(transparent)]
struct ArbitrarySizedWrapper<T, const N: usize>([T; N]);

fn wrap<T>(unwrapped: [T; 10]) -> Wrapper<T> {
    unsafe {
        transmute(unwrapped)
    }
}

fn rewrap<T>(wrapped: Wrapper<T>) -> OtherWrapper<T> {
    unsafe {
        transmute(wrapped)
    }
}

fn unwrap<T>(wrapped: OtherWrapper<T>) -> [T; 10] {
    unsafe {
        transmute(wrapped)
    }
}

fn wrap_arbitrary_size<T, const N: usize>(arr: [T; N]) -> ArbitrarySizedWrapper<T, N> {
    unsafe { transmute(arr) }
}

fn main() {
    let unwrapped = [5_u64; 10];
    let wrapped = wrap(unwrapped);
    assert_eq!([5_u64; 10], wrapped.0);

    let rewrapped = rewrap(wrapped);
    assert_eq!([5_u64; 10], rewrapped.0);

    let unwrapped = unwrap(rewrapped);
    assert_eq!([5_u64; 10], unwrapped);

    let arbitrary_sized_wrapper = wrap_arbitrary_size(unwrapped);
    assert_eq!([5_u64; 10], arbitrary_sized_wrapper.0);
}
