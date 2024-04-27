//@ run-rustfix
#![feature(doc_notable_trait)]

#[doc(spotlight)]
//~^ ERROR unknown `doc` attribute `spotlight`
trait MyTrait {}
