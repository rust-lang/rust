#![feature(staged_api)]
#![feature(rustdoc_internals)]
#![allow(internal_features)]
#![stable(feature = "some_feature", since = "1.3.5")]
#![doc(rust_logo)]

#[stable(feature = "some_feature", since = "1.3.5")]
pub struct Foo {}

impl Foo {
    #[stable(feature = "some_feature", since = "1.3.5")]
    pub fn bar() {}
    #[stable(feature = "some_other_feature", since = "1.3.6")]
    pub fn yo() {}
}
