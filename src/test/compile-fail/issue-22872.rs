// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Wrap<'b> {
    fn foo(&'b mut self);
}

struct Wrapper<P>(P);

impl<'b, P> Wrap<'b> for Wrapper<P>
where P: Process<'b>,
      <P as Process<'b>>::Item: Iterator {
    fn foo(&mut self) {}
}


pub trait Process<'a> {
    type Item;
    fn bar(&'a self);
}

fn push_process<P>(process: P) where P: Process<'static> {
    let _: Box<for<'b> Wrap<'b>> = Box::new(Wrapper(process));
    //~^ ERROR the trait `for<'b> Process<'b>` is not implemented for the type `P` [E0277]
    //~| ERROR the trait `for<'b> core::iter::Iterator` is not implemented for the type
    //~| ERROR cannot infer an appropriate lifetime for lifetime parameter `'b` due to conflicting
}

fn main() {}
