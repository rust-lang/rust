#![deny(invalid_doc_attributes)]
//~^ NOTE defined here
#![doc(x)]
//~^ ERROR unknown `doc` attribute `x`
fn main() {}
