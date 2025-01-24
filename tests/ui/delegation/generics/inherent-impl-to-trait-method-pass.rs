//@ run-pass

#![feature(fn_delegation)]
#![allow(incomplete_features)]

trait Trait<T> {
    fn foo<U>(&self, x: T, y: U) -> (T, U) {
        (x, y)
    }
}

impl<T> Trait<T> for () {}
struct S<T>(T, ());

impl<T> S<T> {
    reuse Trait::foo { self.1 }
}


fn main() {
    let s = S((), ());
    assert_eq!(s.foo(1u32, 2i32), (1u32, 2i32));
}
