// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::rc::RcMut;

trait Foo
{
    fn set(&mut self, v: RcMut<A>);
}

struct B
{
    v: Option<RcMut<A>>
}

impl Foo for B
{
    fn set(&mut self, v: RcMut<A>)
    {
        self.v = Some(v);
    }
}

struct A
{
    v: ~Foo,
}

fn main()
{
    let a = A {v: ~B{v: None} as ~Foo}; //~ ERROR cannot pack type `~B`, which does not fulfill `Send`
    let v = RcMut::new(a); //~ ERROR instantiating a type parameter with an incompatible type
    let w = v.clone();
    v.with_mut_borrow(|p| {p.v.set(w.clone());})
}
