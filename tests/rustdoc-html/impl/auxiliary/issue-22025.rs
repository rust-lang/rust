//@ compile-flags: -Cmetadata=aux

pub mod foo {

    pub trait Foo {}
    pub struct Bar;

    impl Foo for Bar {}

}
