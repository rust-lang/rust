// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub struct Outer<T: Trait>(T);
pub struct Inner<'a> { value: &'a bool }

pub trait Trait {
    type Error;
    fn ready(self) -> Self::Error;
}

impl<'a> Trait for Inner<'a> {
    type Error = Outer<Inner<'a>>;
    fn ready(self) -> Outer<Inner<'a>> { Outer(self) }
}

fn main() {
    let value = true;
    let inner = Inner { value: &value };
    assert_eq!(inner.ready().0.value, &value);
}
