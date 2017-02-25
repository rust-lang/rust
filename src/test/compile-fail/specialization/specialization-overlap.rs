// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(specialization)]

trait Foo { fn foo() {} }
impl<T: Clone> Foo for T {}
impl<T> Foo for Vec<T> {} //~ ERROR E0119

trait Bar { fn bar() {} }
impl<T> Bar for (T, u8) {}
impl<T> Bar for (u8, T) {} //~ ERROR E0119

trait Baz<U> { fn baz() {} }
impl<T> Baz<T> for u8 {}
impl<T> Baz<u8> for T {} //~ ERROR E0119

trait Qux { fn qux() {} }
impl<T: Clone> Qux for T {}
impl<T: Eq> Qux for T {} //~ ERROR E0119

fn main() {}
