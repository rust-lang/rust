//@ run-rustfix
use std::rc::Rc;

pub fn main() {
    let _x = Rc::new(vec![1, 2]).into_iter();
    //~^ ERROR [E0507]
}
