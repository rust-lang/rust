#![crate_name = "foo"]

#[cfg_attr(doc, doc = include_str!("doc.md"))]
pub struct Bar;

mod bar;
