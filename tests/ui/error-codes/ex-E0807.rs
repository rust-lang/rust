//@edition:2018
mod parent{
    pub(in parent) struct Foo;
    //~^ ERROR E0807
    //~| SUGGESTION crate::parent
}
fn main(){}
