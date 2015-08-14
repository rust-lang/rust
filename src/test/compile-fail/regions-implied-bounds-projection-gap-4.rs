// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Along with the other tests in this series, illustrates the
// "projection gap": in this test, we know that `T: 'x`, and that
// is (naturally) enough to conclude that `T: 'x`.

#![feature(rustc_attrs)]
#![allow(dead_code)]
#![allow(unused_variables)]

trait Trait1<'x> {
    type Foo;
}

// calling this fn should trigger a check that the type argument
// supplied is well-formed.
fn wf<T>() { }

fn func<'x, T:Trait1<'x>>(t: &'x T)
{
    wf::<&'x T>();
}

#[rustc_error]
fn main() { } //~ ERROR compilation successful
