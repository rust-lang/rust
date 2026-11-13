//@ run-pass
#![allow(internal_features, unused_braces, unused_parens)]
#![feature(unsized_fn_params)]

struct A<X: ?Sized>(#[allow(dead_code)] X);

fn udrop<T: ?Sized>(_x: T) {}
fn foo() -> Box<[u8]> {
    Box::new(*b"foo")
}
fn afoo() -> Box<A<[u8]>> {
    Box::new(A(*b"foo"))
}

impl std::ops::Add<i32> for A<[u8]> {
    type Output = ();
    fn add(self, _rhs: i32) -> Self::Output {}
}

fn main() {
    udrop::<[u8]>((*foo()));
    *afoo() + 42;
    udrop as fn([u8]);
}
