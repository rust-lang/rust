// check-pass
// compile-flags: -Z chalk

#![allow(dead_code)]

trait Foo { }

struct S<'a, T: ?Sized> where T: Foo {
    data: &'a T,
}

fn bar<T: Foo>(_x: S<'_, T>) { // note that we have an implicit `T: Sized` bound
}

fn main() {
}
