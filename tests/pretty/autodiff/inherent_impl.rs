//@ needs-enzyme

#![feature(autodiff)]
//@ pretty-mode:expanded
//@ pretty-compare-only
//@ pp-exact:inherent_impl.pp

use std::autodiff::autodiff_reverse;

struct Foo {
    a: f64,
}

trait MyTrait {
    fn f(&self, x: f64) -> f64;
    fn df(&self, x: f64, seed: f64) -> (f64, f64);
}

impl MyTrait for Foo {
    #[autodiff_reverse(df, Const, Active, Active)]
    fn f(&self, x: f64) -> f64 {
        self.a * 0.25 * (x * x - 1.0 - 2.0 * x.ln())
    }
}
