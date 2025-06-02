// Ensure that `doc(cfg())` respects `check-cfg`
// Currently not properly working

//@ check-pass
//@ no-auto-check-cfg

//@ revisions: no_check cfg_empty cfg_foo
//@[cfg_empty] compile-flags: --check-cfg cfg()
//@[cfg_foo] compile-flags: --check-cfg cfg(foo)

#![feature(doc_cfg)]
#![doc(cfg(foo))]
//[cfg_empty]~^ WARN unexpected `cfg` condition name: `foo`

#[doc(cfg(foo))]
//[cfg_empty]~^ WARN unexpected `cfg` condition name: `foo`
pub fn foo() {}

#[doc(cfg(foo))]
//[cfg_empty]~^ WARN unexpected `cfg` condition name: `foo`
pub mod module {
    #[allow(unexpected_cfgs)]
    #[doc(cfg(bar))]
    pub fn bar() {}
}
