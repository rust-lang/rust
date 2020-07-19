#![feature(staged_api)]
#![stable(feature = "private_general", since = "1.0.0")]

#[unstable(feature = "private_trait", issue = "none")]
pub trait Bar {}

#[stable(feature = "private_general", since = "1.0.0")]
pub struct Foo {
    // nothing
}

impl Foo {
    #[stable(feature = "private_general", since = "1.0.0")]
    pub fn stable_impl() {}
}

impl Foo {
    #[unstable(feature = "private_trait", issue = "none")]
    pub fn bar() {}

    #[stable(feature = "private_general", since = "1.0.0")]
    pub fn bar2() {}
}

#[stable(feature = "private_general", since = "1.0.0")]
impl Bar for Foo {}
