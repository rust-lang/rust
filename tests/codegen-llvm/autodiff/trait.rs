//@ compile-flags: -Zautodiff=Enable -Zautodiff=NoPostopt -C opt-level=3 -Clto=fat
//@ no-prefer-dynamic
//@ needs-enzyme

// Just check it does not crash for now
// CHECK: ;
#![feature(autodiff)]

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

fn main() {
    let foo = Foo { a: 3.0f64 };
    dbg!(foo.df(1.0, 1.0));
}
