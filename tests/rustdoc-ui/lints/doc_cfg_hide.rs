#![feature(doc_cfg_hide)]

#![doc(cfg_hide = "test")] //~ ERROR
#![doc(cfg_hide)] //~ ERROR

#[doc(cfg_hide(doc))] //~ ERROR
pub fn foo() {}
