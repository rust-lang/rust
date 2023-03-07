#![feature(doc_cfg_hide)]
#![deny(warnings)]

#![doc(cfg_hide = "test")] //~ ERROR
//~^ WARN
#![doc(cfg_hide)] //~ ERROR
//~^ WARN

#[doc(cfg_hide(doc))] //~ ERROR
//~^ WARN
pub fn foo() {}
