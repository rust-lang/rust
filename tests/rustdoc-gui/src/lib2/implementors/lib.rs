#![feature(negative_impls)]

pub trait Whatever {
    type Foo;

    fn method() {}
}

pub struct Struct;
pub struct Struct2;
pub struct Struct3;

impl Whatever for Struct {
    type Foo = u8;
}

impl !Whatever for Struct2 {}

/// fooo
///
/// bar
impl !Whatever for Struct3 {}

impl http::HttpTrait for Struct {}

mod traits {
    pub trait TraitToReexport {
        fn method() {}
    }
}

#[doc(inline)]
pub use traits::TraitToReexport;
