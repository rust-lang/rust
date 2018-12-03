#![feature(unsized_tuple_coercion, unsized_locals)]

struct A<X: ?Sized>(X);

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
    udrop::<(i32, [u8])>((42, *foo()));
    //~^ERROR E0277
    udrop::<A<[u8]>>(A { 0: *foo() });
    //~^ERROR E0277
    udrop::<A<[u8]>>(A(*foo()));
    //~^ERROR E0277
}
