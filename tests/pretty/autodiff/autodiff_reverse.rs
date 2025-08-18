//@ needs-enzyme

#![feature(autodiff)]
//@ pretty-mode:expanded
//@ pretty-compare-only
//@ pp-exact:autodiff_reverse.pp

// Test that reverse mode ad macros are expanded correctly.

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(df1, Duplicated, Const, Active)]
pub fn f1(x: &[f64], y: f64) -> f64 {
    unimplemented!()
}

// Not the most interesting derivative, but who are we to judge
#[autodiff_reverse(df2)]
pub fn f2() {}

#[autodiff_reverse(df3, Duplicated, Const, Active)]
pub fn f3(x: &[f64], y: f64) -> f64 {
    unimplemented!()
}

enum Foo {
    Reverse,
}
use Foo::Reverse;
// What happens if we already have Reverse in type (enum variant decl) and value (enum variant
// constructor) namespace? > It's expected to work normally.
#[autodiff_reverse(df4, Const)]
pub fn f4(x: f32) {
    unimplemented!()
}

#[autodiff_reverse(df5, DuplicatedOnly, Duplicated)]
pub fn f5(x: *const f32, y: &f32) {
    unimplemented!()
}

fn main() {}
