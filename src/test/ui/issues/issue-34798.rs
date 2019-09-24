// run-pass
#![forbid(improper_ctypes)]
#![allow(dead_code)]

#[repr(C)]
pub struct Foo {
    size: u8,
    __value: ::std::marker::PhantomData<i32>,
}

#[repr(C)]
pub struct ZeroSizeWithPhantomData<T>(::std::marker::PhantomData<T>);

#[repr(C)]
pub struct Bar {
    size: u8,
    baz: ZeroSizeWithPhantomData<i32>,
}

extern "C" {
    pub fn bar(_: *mut Foo, _: *mut Bar);
}

fn main() {
}
