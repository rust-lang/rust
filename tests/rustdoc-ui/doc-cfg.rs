#![feature(doc_cfg)]

#[doc(cfg(), cfg(foo, bar))]
//~^ ERROR
//~^^ ERROR
#[doc(cfg(foo), cfg(bar))]
//~^ WARN unexpected `cfg` condition name: `foo`
//~^^ WARN unexpected `cfg` condition name: `bar`
#[doc(cfg())] //~ ERROR
#[doc(cfg(foo, bar))] //~ ERROR
pub fn foo() {}
