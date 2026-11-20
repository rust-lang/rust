#![deny(invalid_doc_attributes)]

#![doc(test(no_crate_inject = 1))]
//~^ ERROR
#![doc(test(attr = 1))]
//~^ ERROR
//~| WARN

#[doc(hidden = true)]
//~^ ERROR
#[doc(hidden("or you will be fired"))]
//~^ ERROR
#[doc(hidden = "handled transparently by codegen")]
//~^ ERROR
#[doc = 1]
//~^ ERROR
//~| WARN
pub struct X;
