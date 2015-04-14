// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #23827

#![feature(core, unboxed_closures)]

pub struct Prototype {
    pub target: u32
}

trait Component {
    fn apply(self, e: u32);
}

impl<C: Component> Fn<(C,)> for Prototype {
    extern "rust-call" fn call(&self, (comp,): (C,)) -> Prototype {
        comp.apply(self.target);
        *self
    }
}

impl<C: Component> FnMut<(C,)> for Prototype {
    extern "rust-call" fn call_mut(&mut self, (comp,): (C,)) -> Prototype {
        Fn::call(*&self, (comp,))
    }
}

impl<C: Component> FnOnce<(C,)> for Prototype {
    //~^ ERROR not all trait items implemented, missing: `Output` [E0046]
    extern "rust-call" fn call_once(self, (comp,): (C,)) -> Prototype {
        Fn::call(&self, (comp,))
    }
}

fn main() {}
