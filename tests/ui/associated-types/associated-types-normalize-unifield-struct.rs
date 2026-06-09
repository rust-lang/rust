//@ run-pass
// Regression test for issue #21010: Normalize associated types in
// various special paths in the `type_is_immediate` function.

pub trait OffsetState: Sized {}
pub trait Offset {
    type State: OffsetState;
    fn dummy(&self) { }
}

#[derive(Copy, Clone)] pub struct X;
impl Offset for X { type State = Y; }

#[derive(Copy, Clone)] pub struct Y;
impl OffsetState for Y {}

pub fn now() -> DateTime<X> { from_utc(Y) }

pub struct DateTime<Off: Offset> { pub offset: Off::State }
pub fn from_utc<Off: Offset>(offset: Off::State) -> DateTime<Off> { DateTime { offset: offset } }

pub fn main() {
    let _x = now();
}
