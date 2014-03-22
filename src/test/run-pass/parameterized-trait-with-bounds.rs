// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(dead_code)];

trait A<T> {}
trait B<T, U> {}
trait C<'a, U> {}

mod foo {
    pub trait D<'a, T> {}
}

fn foo1<T>(_: &A<T>: Send) {}
fn foo2<T>(_: ~A<T>: Send + Share) {}
fn foo3<T>(_: ~B<int, uint>: 'static) {}
fn foo4<'a, T>(_: ~C<'a, T>: 'static + Send) {}
fn foo5<'a, T>(_: ~foo::D<'a, T>: 'static + Send) {}

pub fn main() {}
