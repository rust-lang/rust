#![deny(invalid_doc_attributes)]

#[doc] //~ ERROR valid forms for the attribute are
#[ignore()] //~ ERROR valid forms for the attribute are
//~^ WARN this was previously accepted
#[inline = ""] //~ ERROR valid forms for the attribute are
//~^ WARN this was previously accepted
#[link] //~ ERROR malformed
//~^ WARN attribute should be applied to an `extern` block with non-Rust ABI
//~| WARN previously accepted
#[link = ""] //~ ERROR malformed

fn main() {}
