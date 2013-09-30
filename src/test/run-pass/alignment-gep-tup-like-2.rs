// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Pair<A,B> {
    a: A, b: B
}

struct RecEnum<A>(Rec<A>);
struct Rec<A> {
    val: A,
    rec: Option<@mut RecEnum<A>>
}

fn make_cycle<A:'static>(a: A) {
    let g: @mut RecEnum<A> = @mut RecEnum(Rec {val: a, rec: None});
    g.rec = Some(g);
}

struct Invoker<A,B> {
    a: A,
    b: B,
}

trait Invokable<A,B> {
    fn f(&self) -> (A, B);
}

impl<A:Clone,B:Clone> Invokable<A,B> for Invoker<A,B> {
    fn f(&self) -> (A, B) {
        (self.a.clone(), self.b.clone())
    }
}

fn f<A:Send + Clone + 'static,
     B:Send + Clone + 'static>(
     a: A,
     b: B)
     -> @Invokable<A,B> {
    @Invoker {
        a: a,
        b: b,
    } as @Invokable<A,B>
}

pub fn main() {
    let x = 22_u8;
    let y = 44_u64;
    let z = f(~x, y);
    make_cycle(z);
    let (a, b) = z.f();
    info2!("a={} b={}", *a as uint, b as uint);
    assert_eq!(*a, x);
    assert_eq!(b, y);
}
