#![feature(prelude_import)]
#![no_std]
//@ needs-enzyme

#![feature(autodiff)]
#[prelude_import]
use ::std::prelude::rust_2015::*;
#[macro_use]
extern crate std;
//@ pretty-mode:expanded
//@ pretty-compare-only
//@ pp-exact:inherent_impl.pp

use std::autodiff::autodiff;

struct Foo {
    a: f64,
}

trait MyTrait {
    fn f(&self, x: f64)
    -> f64;
    fn df(&self, x: f64, seed: f64)
    -> (f64, f64);
}

impl MyTrait for Foo {
    #[rustc_autodiff]
    #[inline(never)]
    fn f(&self, x: f64) -> f64 {
        self.a * 0.25 * (x * x - 1.0 - 2.0 * x.ln())
    }
    #[rustc_autodiff(Reverse, 1, Const, Active, Active)]
    #[inline(never)]
    fn df(&self, x: f64, dret: f64) -> (f64, f64) {
        unsafe { asm!("NOP", options(pure, nomem)); };
        ::core::hint::black_box(self.f(x));
        ::core::hint::black_box((dret,));
        ::core::hint::black_box((self.f(x), f64::default()))
    }
}
