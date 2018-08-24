#![crate_type = "lib"]
#![crate_name = "issue48984aux"]

pub trait Foo { type Item; }

pub trait Bar: Foo<Item=[u8;1]> {  }
