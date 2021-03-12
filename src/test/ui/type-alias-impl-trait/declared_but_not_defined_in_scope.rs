// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

fn main() {}

mod boo {
    // declared in module but not defined inside of it
    pub type Boo = impl ::std::fmt::Debug; //~ ERROR could not find defining uses
}

fn bomp() -> boo::Boo {
    ""
    //~^ mismatched types
}
