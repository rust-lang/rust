use std::rc::Rc;
pub struct Foo;

pub type Function = Rc<Foo>;

impl Function {}
//~^ ERROR cannot define inherent `impl` for a type outside of the crate where the type is defined [E0116]
fn main(){}
