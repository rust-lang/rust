#![feature(type_alias_impl_trait)]

fn main() {}

mod boo {
    // declared in module but not defined inside of it
    pub type Boo = impl ::std::fmt::Debug; //~ ERROR could not find defining uses
}

fn bomp() -> boo::Boo {
    ""
    //~^ mismatched types
}
