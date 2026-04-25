//@ check-pass

#![feature(const_closures, const_destruct, const_trait_impl)]

use std::marker::Destruct;
use std::num::NonZero;

const trait T {
    fn a(&mut self, f: impl [const] Fn() + [const] Destruct);
    fn b(&mut self);
}

struct S;

impl const T for S {
    fn a(&mut self, f: impl [const] Fn() + [const] Destruct) {
        f()
    }

    fn b(&mut self) {
        self.a(const || {});
    }
}

fn main() {}
