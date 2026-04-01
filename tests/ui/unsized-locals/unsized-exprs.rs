#![feature(unsized_fn_params)]

struct A<X: ?Sized>(X);

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
    udrop::<A<[u8]>>(A { 0: *foo() });
    //~^ERROR E0277
    udrop::<A<[u8]>>(A(*foo()));
    //~^ERROR E0277
}
