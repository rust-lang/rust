use std::rc::Rc;

#[derive(PartialEq)] //~ NOTE: in this expansion
pub struct Function {
    callback: Rc<dyn Fn()>, //~ ERROR: can't compare `Rc<dyn Fn()>`
}

fn main() {}
