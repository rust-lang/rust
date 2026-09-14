//@ compile-flags: -Zautodiff=Enable -Zautodiff=NoPostopt -C opt-level=3 -Clto=fat
//@ no-prefer-dynamic
//@ needs-enzyme

// Just check it does not crash for now
// CHECK: ;
#![feature(autodiff)]
#![feature(core_intrinsics)]
#![feature(rustc_attrs)]

use std::autodiff::autodiff_reverse;

struct Foo {
    a: f64,
}

trait MyTrait {
    #[autodiff_reverse(df, Const, Active, Active)]
    fn f(&self, x: f64) -> f64;
}

impl MyTrait for Foo {
    fn f(&self, x: f64) -> f64 {
        x.sin()
    }
}

fn main() {
    let foo = Foo { a: 3.0f64 };
    dbg!(foo.df(2.0, 1.0));
    dbg!(2.0_f64.cos());
}
