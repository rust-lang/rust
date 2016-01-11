// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// check that we don't go mad when there are errors in trait evaluation

use std::marker::PhantomData;

pub trait Foo { type Output: 'static; }
pub trait Bar<P> {}
pub trait Baz<P> {}
pub struct Qux<T>(PhantomData<*mut T>);

impl<P, T: Baz<P>> Bar<P> for Option<T> {}
impl<T> Bar<*mut T> for Option<Qux<T>> {}

impl<T: 'static, W: Foo<Output=T>> Baz<*mut T> for W {}

fn main() {}
