#![deny(invalid_doc_attributes)]

#[doc(primitive = "foo")]
//~^ ERROR unknown `doc` attribute `primitive`
//~| WARN
mod bar {}

fn main() {}
