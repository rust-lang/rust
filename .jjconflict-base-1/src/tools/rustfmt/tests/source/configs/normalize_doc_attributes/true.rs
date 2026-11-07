// rustfmt-normalize_doc_attributes: true
// Normalize doc attributes

#![doc = " Example documentation"]

#[doc = " Example item documentation"]
pub enum Foo {}

#[doc = "        Lots of space"]
pub enum Bar {}

#[doc = "no leading space"]
pub mod FooBar {}
