// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(specialization)]

fn main() {}

pub trait Alpha<T> { }

pub trait Beta {
    type Event;
}

pub trait Delta {
    type Handle;
    fn process(&self);
}

pub struct Parent<A, T>(A, T);

impl<A, T> Delta for Parent<A, T>
where A: Alpha<T::Handle>,
      T: Delta,
      T::Handle: Beta<Event = <Handle as Beta>::Event> {
    type Handle = Handle;
    default fn process(&self) {
        unimplemented!()
    }
}

impl<A, T> Delta for Parent<A, T>
where A: Alpha<T::Handle> + Alpha<Handle>,
      T: Delta,
      T::Handle: Beta<Event = <Handle as Beta>::Event> {
      fn process(&self) {
        unimplemented!()
      }
}

pub struct Handle;

impl Beta for Handle {
    type Event = ();
}
