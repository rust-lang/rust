// run-pass
#![allow(unused_allocation)]

use std::rc::Rc;

trait Trait {
    fn trait_method<'a>(self: &'a Box<Rc<Self>>) -> &'a [i32];
}

impl Trait for Vec<i32> {
    fn trait_method<'a>(self: &'a Box<Rc<Self>>) -> &'a [i32] {
        &***self
    }
}

fn main() {
    let v = vec![1, 2, 3];

    assert_eq!(&[1, 2, 3], Box::new(Rc::new(v)).trait_method());
}
