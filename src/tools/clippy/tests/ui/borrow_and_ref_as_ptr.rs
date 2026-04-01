// Make sure that `ref_as_ptr` is not emitted when `borrow_as_ptr` is.

#![warn(clippy::ref_as_ptr, clippy::borrow_as_ptr)]

fn f<T>(_: T) {}

fn main() {
    let mut val = 0;
    f(&val as *const _);
    //~^ borrow_as_ptr
    f(&mut val as *mut i32);
    //~^ borrow_as_ptr
}
