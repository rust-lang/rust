//@ run-rustfix
#![feature(doc_notable_trait)]
#![deny(invalid_doc_attributes)]

#[doc(spotlight)]
//~^ ERROR unknown `doc` attribute `spotlight`
trait MyTrait {}
