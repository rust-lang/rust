#[doc(hidden)]
pub mod error {
    pub struct Foo;

    pub trait HiddenTrait {
        fn hidden(&self) {}
    }
}

pub struct Bar;
pub trait PubTrait {
    fn public(&self) {}
}

impl crate::error::HiddenTrait for crate::error::Foo {}
impl crate::error::HiddenTrait for Bar {}

impl PubTrait for crate::error::Foo {}
impl PubTrait for Bar {}
