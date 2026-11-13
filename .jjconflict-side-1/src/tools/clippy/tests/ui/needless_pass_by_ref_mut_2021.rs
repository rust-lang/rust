//@edition: 2021
//@check-pass
#![warn(clippy::needless_pass_by_ref_mut)]

struct Data<T: ?Sized> {
    value: T,
}

// Unsafe functions should not warn.
unsafe fn get_mut_unchecked<T>(ptr: &mut std::ptr::NonNull<Data<T>>) -> &mut T {
    &mut (*ptr.as_ptr()).value
}
