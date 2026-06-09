//@ revisions: nightly stable
//@[nightly] only-nightly
//@[stable] only-stable

// This checks that without enabling the autodiff feature, we can't import std::autodiff::autodiff;

#![crate_type = "lib"]

use std::autodiff::autodiff_reverse;
//[nightly]~^ ERROR use of unstable library feature `autodiff`
//[stable]~^^ ERROR use of unstable library feature `autodiff`
//[stable]~| NOTE see issue #124509 <https://github.com/rust-lang/rust/issues/124509> for more information
//[stable]~| HELP add `#![feature(autodiff)]` to the crate attributes to enable
//[stable]~| NOTE this compiler was built on YYYY-MM-DD; consider upgrading it if it is out of date

#[autodiff_reverse(dfoo)]
//[nightly]~^ ERROR use of unstable library feature `autodiff` [E0658]
//[stable]~^^ ERROR use of unstable library feature `autodiff` [E0658]
//[stable]~| NOTE see issue #124509 <https://github.com/rust-lang/rust/issues/124509> for more information
//[stable]~| HELP add `#![feature(autodiff)]` to the crate attributes to enable
//[stable]~| NOTE this compiler was built on YYYY-MM-DD; consider upgrading it if it is out of date
fn foo() {}
