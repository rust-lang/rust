// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test a scenario where we generate a constraint like `?1 <: &?2`.
// In such a case, it is important that we instantiate `?1` with `&?3`
// where `?3 <: ?2`, and not with `&?2`. This is a regression test for
// #18653. The important thing is that we build.

use std::cell::RefCell;

enum Wrap<A> {
    WrapSome(A),
    WrapNone
}

use Wrap::*;

struct T;
struct U;

trait Get<T: ?Sized> {
    fn get(&self) -> &T;
}

impl Get<MyShow + 'static> for Wrap<T> {
    fn get(&self) -> &(MyShow + 'static) {
        static x: usize = 42;
        &x
    }
}

impl Get<usize> for Wrap<U> {
    fn get(&self) -> &usize {
        static x: usize = 55;
        &x
    }
}

trait MyShow { fn dummy(&self) { } }
impl<'a> MyShow for &'a (MyShow + 'a) { }
impl MyShow for usize { }
fn constrain<'a>(rc: RefCell<&'a (MyShow + 'a)>) { }

fn main() {
    let mut collection: Wrap<_> = WrapNone;

    {
        let __arg0 = Get::get(&collection);
        let __args_cell = RefCell::new(__arg0);
        constrain(__args_cell);
    }
    collection = WrapSome(T);
}
