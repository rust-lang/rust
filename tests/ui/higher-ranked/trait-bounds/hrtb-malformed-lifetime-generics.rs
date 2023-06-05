// Test that Fn-family traits with lifetime parameters shouldn't compile and
// we suggest the usage of higher-rank trait bounds instead.

fn fa(_: impl Fn<'a>(&'a str) -> bool) {}
//~^ ERROR `Fn` traits cannot take lifetime parameters

fn fb(_: impl FnMut<'a, 'b>(&'a str, &'b str) -> bool) {}
//~^ ERROR `Fn` traits cannot take lifetime parameters

fn fc(_: impl std::fmt::Display + FnOnce<'a>(&'a str) -> bool + std::fmt::Debug) {}
//~^ ERROR `Fn` traits cannot take lifetime parameters

use std::ops::Fn as AliasedFn;
fn fd(_: impl AliasedFn<'a>(&'a str) -> bool) {}
//~^ ERROR `Fn` traits cannot take lifetime parameters

fn fe<F>(_: F) where F: Fn<'a>(&'a str) -> bool {}
//~^ ERROR `Fn` traits cannot take lifetime parameters

fn main() {}
