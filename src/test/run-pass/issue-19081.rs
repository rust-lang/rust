// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-pretty -- FIXME(#17362) pretty prints as `Hash<<Self as Hasher...` which fails to parse

pub trait Hasher {
    type State;

    fn hash<T: Hash<
        <Self as Hasher>::State
    >>(&self, value: &T) -> u64;
}

pub trait Hash<S> {
    fn hash(&self, state: &mut S);
}

fn main() {}
