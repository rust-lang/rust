// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test for the outlives relation when applied to a projection on a
// type with bound regions. In this case, we are checking that
// `<for<'r> fn(&'r T) as TheTrait>::TheType: 'a` If we're not
// careful, we could wind up with a constraint that `'r:'a`, but since
// `'r` is bound, that leads to badness. This test checks that
// everything works.

#![feature(rustc_attrs)]
#![allow(dead_code)]

trait TheTrait {
    type TheType;
}

fn wf<T>() { }

type FnType<T> = for<'r> fn(&'r T);

fn foo<'a,'b,T>()
    where FnType<T>: TheTrait
{
    wf::< <FnType<T> as TheTrait>::TheType >();
}

#[rustc_error]
fn main() { } //~ ERROR compilation successful
