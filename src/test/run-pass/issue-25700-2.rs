// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub trait Parser {
    type Input;
}

pub struct Iter<P: Parser>(P, P::Input);

pub struct Map<P, F>(P, F);
impl<P, F> Parser for Map<P, F> where F: FnMut(P) {
    type Input = u8;
}

trait AstId { type Untyped; }
impl AstId for u32 { type Untyped = u32; }

fn record_type<Id: AstId>(i: Id::Untyped) -> u8 {
    Iter(Map(i, |_: Id::Untyped| {}), 42).1
}

pub fn main() {
   assert_eq!(record_type::<u32>(3), 42);
}
