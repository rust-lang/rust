// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(managed_boxes)];

struct pair<A,B> {
    a: A, b: B
}

trait Invokable<A> {
    fn f(&self) -> (A, u16);
}

struct Invoker<A> {
    a: A,
    b: u16,
}

impl<A:Clone> Invokable<A> for Invoker<A> {
    fn f(&self) -> (A, u16) {
        (self.a.clone(), self.b)
    }
}

fn f<A:Clone + 'static>(a: A, b: u16) -> @Invokable<A> {
    @Invoker {
        a: a,
        b: b,
    } as @Invokable<A>
}

pub fn main() {
    let (a, b) = f(22_u64, 44u16).f();
    info!("a={:?} b={:?}", a, b);
    assert_eq!(a, 22u64);
    assert_eq!(b, 44u16);
}
