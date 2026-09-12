//@edition:2018
mod a{
    pub(in a) struct Foo;
    //~^ ERROR E0807
    //~| SUGGESTION crate::a
    pub(in parent) struct Baz;
    //~^ ERROR E0807
    //~| SUGGESTION super
    //~| SUGGESTION crate::parent
}

fn main(){}
