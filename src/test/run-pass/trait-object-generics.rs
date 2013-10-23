// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(managed_boxes)];

// test for #8664

pub trait Trait2<A> {
    fn doit(&self);
}

pub struct Impl<A1, A2, A3> {
    /*
     * With A2 we get the ICE:
     * task <unnamed> failed at 'index out of bounds: the len is 1 but the index is 1', /home/tortue/rust_compiler_newest/src/librustc/middle/subst.rs:58
     */
    t: ~Trait2<A2>
}

impl<A1, A2, A3> Impl<A1, A2, A3> {
    pub fn step(&self) {
        self.t.doit()
    }
}

// test for #8601

enum Type<T> { Constant }

trait Trait<K,V> {
    fn method(&self,Type<(K,V)>) -> int;
}

impl<V> Trait<u8,V> for () {
    fn method(&self, _x: Type<(u8,V)>) -> int { 0 }
}

pub fn main () {
    let a = @() as @Trait<u8, u8>;
    assert_eq!(a.method(Constant), 0);
}
