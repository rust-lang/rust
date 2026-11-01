use std::rc::Rc;

#[derive(PartialEq)] //~ NOTE in this expansion
pub struct Function {
    callback: Rc<dyn Fn()>, //~ ERROR binary operation `==` cannot be applied to type `Rc<dyn Fn()>`
}

fn main() {}
