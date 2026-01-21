#![feature(doc_cfg)]

#[doc(cfg(foo), cfg(bar))]
//~^ WARN unexpected `cfg` condition name: `foo`
//~| WARN unexpected `cfg` condition name: `bar`
#[doc(auto_cfg(42))] //~ ERROR
#[doc(auto_cfg(hide(true)))] //~ ERROR
#[doc(auto_cfg(hide(42)))] //~ ERROR
#[doc(auto_cfg(hide("a")))] //~ ERROR
#[doc(auto_cfg = 42)] //~ ERROR
#[doc(auto_cfg = "a")] //~ ERROR
// Shouldn't lint
#[doc(auto_cfg(hide(windows)))]
#[doc(auto_cfg(hide(feature = "windows")))]
//~^ WARN unexpected `cfg` condition name: `feature`
#[doc(auto_cfg(hide(foo)))]
//~^ WARN unexpected `cfg` condition name: `foo`
pub fn foo() {}
