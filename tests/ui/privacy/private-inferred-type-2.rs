//@ aux-build:private-inferred-type.rs
#![allow(private_interfaces)]

extern crate private_inferred_type as ext;

mod m {
    struct Priv;
    pub struct Pub<T>(pub T);

    impl Pub<Priv> {
        pub fn get_priv() -> Priv { Priv }
        pub fn static_method() {}
    }
}

fn main() {
    m::Pub::get_priv; //~ ERROR type `Priv` is private
    m::Pub::static_method; //~ ERROR type `Priv` is private
    ext::Pub::static_method; //~ ERROR type `ext::Priv` is private
}
