#![deny(invalid_doc_attributes)]
#![feature(doc_cfg)]

#[doc(cfg(), cfg(foo, bar))]
//~^ ERROR
//~| ERROR
#[doc(cfg())] //~ ERROR
#[doc(cfg(foo, bar))] //~ ERROR
#[doc(auto_cfg(hide(foo::bar)))]
pub fn foo() {}
