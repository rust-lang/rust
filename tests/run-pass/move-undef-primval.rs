#![allow(deprecated)]

struct Foo {
    _inner: i32,
}

fn main() {
    unsafe {
        let foo = Foo {
            _inner: std::mem::uninitialized(),
        };
        let _bar = foo;
    }
}
