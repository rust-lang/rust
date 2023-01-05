#![deny(invalid_doc_attributes)]
//~^ NOTE defined here
#![doc(x)]
//~^ ERROR unknown `doc` attribute `x`
//~| WARNING will become a hard error
//~| NOTE see issue #82730
fn main() {}
