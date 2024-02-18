//@ revisions: current negative
#![feature(specialization)]
#![cfg_attr(negative, feature(with_negative_coherence))]
#![allow(incomplete_features)]

trait X {}
trait Y: X {}
trait Z {
    type Assoc: Y;
}
struct A<T>(T);

impl<T> Y for T where T: X {}
impl<T: X> Z for A<T> {
    type Assoc = T;
}

// this impl is invalid, but causes an ICE anyway
impl<T> From<<A<T> as Z>::Assoc> for T {}
//~^ ERROR type parameter `T` must be used as the type parameter for some local type (e.g., `MyStruct<T>`)

fn main() {}
