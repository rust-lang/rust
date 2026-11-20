#![crate_name = "realcore"]
#![feature(staged_api)]
#![unstable(feature = "extremely_unstable", issue = "none")]

#[unstable(feature = "extremely_unstable_foo", issue = "none")]
pub struct Foo {}

#[unstable(feature = "extremely_unstable_foo", issue = "none")]
pub trait Join {}

#[unstable(feature = "extremely_unstable_foo", issue = "none")]
impl Join for Foo {}

#[stable(feature = "faked_deref", since = "1.47.0")]
pub trait Deref {}
