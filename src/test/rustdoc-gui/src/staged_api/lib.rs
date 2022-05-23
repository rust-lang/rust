#![feature(staged_api)]
#![stable(feature = "some_feature", since = "1.3.5")]

#[stable(feature = "some_feature", since = "1.3.5")]
pub struct Foo {}

impl Foo {
    #[stable(feature = "some_feature", since = "1.3.5")]
    pub fn bar() {}
}
