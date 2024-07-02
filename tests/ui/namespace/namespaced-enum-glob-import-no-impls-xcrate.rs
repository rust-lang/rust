//@ aux-build:namespaced_enums.rs
extern crate namespaced_enums;

mod m {
    pub use namespaced_enums::Foo::*;
}

pub fn main() {
    use namespaced_enums::Foo::*;

    foo(); //~ ERROR cannot find function `foo`
    m::foo(); //~ ERROR cannot find function `foo`
    bar(); //~ ERROR cannot find function `bar`
    m::bar(); //~ ERROR cannot find function `bar`
}
