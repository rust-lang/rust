use std::any::Any;

struct Something<'a> {
    broken: Box<dyn Any + 'a> //~ ERROR lifetime bound not satisfied
}
fn main() {}
