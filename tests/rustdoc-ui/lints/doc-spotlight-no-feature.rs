#![deny(invalid_doc_attributes)]

#[doc(spotlight)]
//~^ ERROR unknown `doc` attribute `spotlight`
pub trait MyTrait {}
