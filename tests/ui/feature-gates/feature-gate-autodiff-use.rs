//@ revisions: nightly stable
//@[nightly] only-nightly
//@[stable] act-as-stable

// This checks that without enabling the autodiff feature, we can't import std::autodiff::autodiff;

#![crate_type = "lib"]

use std::autodiff::autodiff_reverse;
//[stable]~^ ERROR use of unstable library feature `autodiff`
//[nightly]~^^ ERROR use of unstable library feature `autodiff`
//[nightly]~| NOTE see issue #124509 <https://github.com/rust-lang/rust/issues/124509> for more information
//[nightly]~| HELP add `#![feature(autodiff)]` to the crate attributes to enable
//[nightly]~| NOTE this compiler was built on YYYY-MM-DD; consider upgrading it if it is out of date

#[autodiff_reverse(dfoo)]
//[stable]~^ ERROR use of unstable library feature `autodiff` [E0658]
//[nightly]~^^ ERROR use of unstable library feature `autodiff` [E0658]
//[nightly]~| NOTE see issue #124509 <https://github.com/rust-lang/rust/issues/124509> for more information
//[nightly]~| HELP add `#![feature(autodiff)]` to the crate attributes to enable
//[nightly]~| NOTE this compiler was built on YYYY-MM-DD; consider upgrading it if it is out of date
fn foo() {}
