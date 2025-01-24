// Make sure that `ref_as_ptr` is not emitted when `borrow_as_ptr` is.

#![warn(clippy::ref_as_ptr, clippy::borrow_as_ptr)]

fn f<T>(_: T) {}

fn main() {
    let mut val = 0;
    f(&val as *const _);
    f(&mut val as *mut i32);
}
