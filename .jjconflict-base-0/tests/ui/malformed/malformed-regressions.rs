#![deny(invalid_doc_attributes)]

#[doc] //~ ERROR valid forms for the attribute are
#[ignore()] //~ ERROR valid forms for the attribute are
//~^ WARN this was previously accepted
#[inline = ""] //~ ERROR valid forms for the attribute are
//~^ WARN this was previously accepted
#[link] //~ ERROR malformed
//~^ WARN attribute cannot be used on
//~| WARN previously accepted
#[link = ""] //~ ERROR malformed
//~^ WARN attribute cannot be used on
//~| WARN previously accepted

fn main() {}
