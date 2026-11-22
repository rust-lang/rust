#![doc(test(no_crate_inject = 1))]
//~^ ERROR
#![doc(test(attr = 1))]
//~^ ERROR

#[doc(hidden = true)]
//~^ ERROR
#[doc(hidden("or you will be fired"))]
//~^ ERROR
#[doc(hidden = "handled transparently by codegen")]
//~^ ERROR
#[doc = 1]
//~^ ERROR
pub struct X;
