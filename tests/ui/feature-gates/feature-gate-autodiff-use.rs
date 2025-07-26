//@ revisions: has_support no_support
//@[no_support] ignore-enzyme
//@[has_support] needs-enzyme

// This checks that without enabling the autodiff feature, we can't import std::autodiff::autodiff;

#![crate_type = "lib"]

use std::autodiff::autodiff_reverse;
//[has_support]~^ ERROR use of unstable library feature `autodiff`
//[no_support]~^^ ERROR use of unstable library feature `autodiff`

#[autodiff_reverse(dfoo)]
//[has_support]~^ ERROR use of unstable library feature `autodiff` [E0658]
//[no_support]~^^ ERROR use of unstable library feature `autodiff` [E0658]
//[no_support]~| ERROR this rustc version does not support autodiff
fn foo() {}
