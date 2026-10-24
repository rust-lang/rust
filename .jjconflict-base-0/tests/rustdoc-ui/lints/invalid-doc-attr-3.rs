#![deny(invalid_doc_attributes)]

#![doc(test(no_crate_inject = 1))]
//~^ ERROR
//~| WARN
#![doc(test(attr = 1))]
//~^ ERROR
//~| WARN

#[doc(hidden = true)]
//~^ ERROR
//~| WARN
#[doc(hidden("or you will be fired"))]
//~^ ERROR
//~| WARN
#[doc(hidden = "handled transparently by codegen")]
//~^ ERROR
//~| WARN
#[doc = 1]
//~^ ERROR
//~| WARN
pub struct X;
