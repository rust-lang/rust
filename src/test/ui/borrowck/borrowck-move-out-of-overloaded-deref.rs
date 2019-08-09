use std::rc::Rc;

pub fn main() {
    let _x = *Rc::new("hi".to_string());
    //~^ ERROR cannot move out of an `Rc`
}
