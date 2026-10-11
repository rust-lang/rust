#![deny(invalid_doc_attributes)]

#[doc(primitive = "foo")]
//~^ ERROR reserved `doc` attribute `primitive`
mod bar {}

fn main() {}
