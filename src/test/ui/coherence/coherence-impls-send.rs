// revisions: old re

#![cfg_attr(re, feature(re_rebalance_coherence))]
#![feature(optin_builtin_traits)]
#![feature(overlapping_marker_traits)]

use std::marker::Copy;

enum TestE {
  A
}

struct MyType;

struct NotSync;
impl !Sync for NotSync {}

unsafe impl Send for TestE {}
unsafe impl Send for MyType {}
unsafe impl Send for (MyType, MyType) {}
//[old]~^ ERROR E0117
//[re]~^^ ERROR E0117

unsafe impl Send for &'static NotSync {}
//[old]~^ ERROR E0321
//[re]~^^ ERROR E0321

unsafe impl Send for [MyType] {}
//[old]~^ ERROR E0117
//[re]~^^ ERROR E0117

unsafe impl Send for &'static [NotSync] {}
//[old]~^ ERROR E0117
//[re]~^^ ERROR E0117

fn main() {
}
