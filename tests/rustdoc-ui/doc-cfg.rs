#![feature(doc_cfg)]

#[doc(cfg(), cfg(foo, bar))]
//~^ ERROR
//~^^ ERROR
#[doc(cfg(foo), cfg(bar))]
//~^ WARN unexpected `cfg` condition name: `foo`
//~^^ WARN unexpected `cfg` condition name: `bar`
#[doc(cfg())] //~ ERROR
#[doc(cfg(foo, bar))] //~ ERROR
#[doc(auto_cfg(42))] //~ ERROR
#[doc(auto_cfg(hide(true)))] //~ ERROR
#[doc(auto_cfg(hide(42)))] //~ ERROR
#[doc(auto_cfg(hide("a")))] //~ ERROR
#[doc(auto_cfg(hide(foo::bar)))] //~ ERROR
#[doc(auto_cfg = 42)] //~ ERROR
#[doc(auto_cfg = "a")] //~ ERROR
// Shouldn't lint
#[doc(auto_cfg(hide(windows)))]
#[doc(auto_cfg(hide(feature = "windows")))]
#[doc(auto_cfg(hide(foo)))]
pub fn foo() {}
