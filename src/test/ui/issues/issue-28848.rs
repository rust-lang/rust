// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-compare-mode-nll

struct Foo<'a, 'b: 'a>(&'a &'b ());

impl<'a, 'b> Foo<'a, 'b> {
    fn xmute(a: &'b ()) -> &'a () {
        unreachable!()
    }
}

pub fn foo<'a, 'b>(u: &'b ()) -> &'a () {
    Foo::<'a, 'b>::xmute(u) //~ ERROR lifetime bound not satisfied
}

fn main() {}
