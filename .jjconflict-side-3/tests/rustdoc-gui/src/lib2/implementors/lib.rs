#![feature(negative_impls)]

pub trait Whatever {
    type Foo;

    fn method() {}
}

pub struct Struct;
pub struct Struct2;

impl Whatever for Struct {
    type Foo = u8;
}

impl !Whatever for Struct2 {}

impl http::HttpTrait for Struct {}

mod traits {
    pub trait TraitToReexport {
        fn method() {}
    }
}

#[doc(inline)]
pub use traits::TraitToReexport;
