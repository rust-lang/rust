// run-pass

#![feature(const_ptr_nonnull)]

use std::ptr::NonNull;

const fn dangling() -> NonNull<u32> {
    NonNull::dangling()
}

const fn cast<T, U>(non_null: NonNull<T>) -> NonNull<U> {
    non_null.cast()
}

pub fn main() {
    assert_eq!(dangling(), NonNull::dangling());

    let mut i: i32 = 10;
    let non_null_t = NonNull::new(&mut i).unwrap();
    let non_null_u: NonNull<u32> = cast(non_null_t);
    assert_eq!(non_null_t.as_ptr(), non_null_u.as_ptr() as *mut i32);
}
