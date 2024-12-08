//@ run-pass
#![allow(incomplete_features, unused_braces, unused_parens)]
#![feature(unsized_tuple_coercion, unsized_locals, unsized_fn_params)]

struct A<X: ?Sized>(#[allow(dead_code)] X);

fn udrop<T: ?Sized>(_x: T) {}
fn foo() -> Box<[u8]> {
    Box::new(*b"foo")
}
fn tfoo() -> Box<(i32, [u8])> {
    Box::new((42, *b"foo"))
}
fn afoo() -> Box<A<[u8]>> {
    Box::new(A(*b"foo"))
}

impl std::ops::Add<i32> for A<[u8]> {
    type Output = ();
    fn add(self, _rhs: i32) -> Self::Output {}
}

fn main() {
    udrop::<[u8]>(loop {
        break *foo();
    });
    udrop::<[u8]>(if true { *foo() } else { *foo() });
    udrop::<[u8]>({ *foo() });
    udrop::<[u8]>((*foo()));
    udrop::<[u8]>((*tfoo()).1);
    *afoo() + 42;
    udrop as fn([u8]);
}
