struct A;
impl A<B> {}
//~^ ERROR cannot find type `B` in this scope
//~| ERROR struct takes 0 generic arguments but 1 generic argument was supplied
fn main() {}
