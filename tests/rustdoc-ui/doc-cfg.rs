#![feature(doc_cfg)]

#[doc(cfg(), cfg(foo, bar))]
//~^ ERROR malformed `doc` attribute input
//~| ERROR malformed `doc` attribute input
#[doc(cfg())] //~ ERROR
#[doc(cfg(foo, bar))] //~ ERROR
#[doc(auto_cfg(hide(foo::bar)))] //~ ERROR
pub fn foo() {}
