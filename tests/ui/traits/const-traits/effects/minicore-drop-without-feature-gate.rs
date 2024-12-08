//@ aux-build:minicore.rs
//@ compile-flags: --crate-type=lib -Znext-solver
//@ revisions: yes no
//@[yes] check-pass
// gate-test-const_destruct

#![feature(no_core, const_trait_impl)]
#![cfg_attr(yes, feature(const_destruct))]
#![no_std]
#![no_core]

extern crate minicore;
use minicore::*;

struct ConstDrop;
impl const Drop for ConstDrop {
    fn drop(&mut self) {}
}

// Make sure that `ConstDrop` can only be dropped when the `const_drop`
// feature gate is enabled. Otherwise, we should error if there is a drop
// impl at all.
const fn test() {
    let _ = ConstDrop;
    //[no]~^ ERROR destructor of `ConstDrop` cannot be evaluated at compile-time
}
