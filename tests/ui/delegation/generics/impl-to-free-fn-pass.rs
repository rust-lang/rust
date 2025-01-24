//@ run-pass
#![feature(fn_delegation)]
#![allow(incomplete_features)]

mod to_reuse {
    pub fn foo<T, U>(_: T, y: U) -> U { y }
}

trait Trait<T> {
    fn foo(&self, x: T) -> T { x }
}
struct F;
impl<T> Trait<T> for F {}

struct S<T>(F, T);

impl<T, U> Trait<T> for S<U> {
    reuse to_reuse::foo { &self.0 }
}

impl<T> S<T> {
    reuse to_reuse::foo;
}

fn main() {
    let s = S(F, 42);
    assert_eq!(S::<i32>::foo(F, 1), 1);
    assert_eq!(<S<_> as Trait<_>>::foo(&s, 1), 1);
}
