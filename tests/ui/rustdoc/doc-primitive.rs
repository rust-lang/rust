#![deny(invalid_doc_attributes)]

#[doc(primitive = "foo")]
//~^ ERROR unknown `doc` attribute `primitive`
mod bar {}

fn main() {}
