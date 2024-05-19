//@ compile-flags: -Znext-solver
//@ check-pass

use std::fmt::Display;
use std::rc::Rc;

fn mk<T: ?Sized>(t: Option<&T>) -> Rc<T> {
    todo!()
}

fn main() {
    let mut x = None;
    let y = mk(x);
    // Don't treat the line below as a unsize coercion `Rc<?0> ~> Rc<dyn Display>`
    let z: Rc<dyn Display> = y;
    x = Some(&1 as &dyn Display);
}
