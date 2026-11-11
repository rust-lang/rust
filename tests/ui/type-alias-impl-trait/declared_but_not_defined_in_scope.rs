#![feature(type_alias_impl_trait)]

fn main() {}

mod boo {
    // declared in module but not defined inside of it
    pub type Boo = impl ::std::fmt::Debug; //~ ERROR unconstrained opaque type
}

fn bomp() -> boo::Boo {
    ""
    //~^ ERROR mismatched types
}
