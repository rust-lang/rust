#![feature(dropck_eyepatch)]

// This test checks the `#[may_dangle]` attribute syntax.

struct Foo<T>(*const T);

// No error: "drops" bound
unsafe impl<#[may_dangle] T> Drop for Foo<T> {
    fn drop(&mut self) { }
}

struct Bar<T>(*const T);

// No error: "borrows" bound
unsafe impl<#[may_dangle(borrow)] T> Drop for Bar<T> {
    fn drop(&mut self) { }
}

struct Baz<T>(*const T);

// Error: invalid syntax
unsafe impl<#[may_dangle(invalid)] T> Drop for Baz<T> {
    fn drop(&mut self) { }
}

pub fn main() { }
