//@ run-rustfix
#![allow(dead_code)]
trait O {
    type M;
}
trait U<A: O> {
    const N: A::M;
}
impl<D> O for D {
    type M = u8;
}
impl<C: O> U<C> for u16 {
    const N: C::M = 4u8; //~ ERROR mismatched types
}
fn main() {}
