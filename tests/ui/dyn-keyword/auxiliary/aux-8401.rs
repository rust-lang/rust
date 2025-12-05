// for this issue, this code must be built in a library

use std::mem;

trait A {
    fn dummy(&self) { }
}
struct B;
impl A for B {}

fn bar<T>(_: &mut dyn A, _: &T) {}

fn foo<T>(t: &T) {
    let mut b = B;
    bar(&mut b as &mut dyn A, t)
}
