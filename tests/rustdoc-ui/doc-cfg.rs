#![feature(doc_cfg)]

#[doc(cfg(), cfg(foo, bar))]
//~^ ERROR
//~^^ ERROR
#[doc(cfg(foo), cfg(bar))] // ok!
#[doc(cfg())] //~ ERROR
#[doc(cfg(foo, bar))] //~ ERROR
pub fn foo() {}
