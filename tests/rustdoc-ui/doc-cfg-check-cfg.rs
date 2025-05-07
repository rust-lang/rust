// Ensure that `doc(cfg())` respects `check-cfg`
// Currently not properly working
#![feature(doc_cfg)]
#![deny(unexpected_cfgs)]

//@revisions: no_check cfg_empty cfg_foo
//@[cfg_empty] compile-flags: --check-cfg cfg()
//@[cfg_foo] compile-flags: --check-cfg cfg(foo)

//@[no_check] check-pass
//@[cfg_empty] check-pass
//@[cfg_empty] known-bug: #138358
//@[cfg_foo] check-pass

#[doc(cfg(foo))]
pub fn foo() {}
