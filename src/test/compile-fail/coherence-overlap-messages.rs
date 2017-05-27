// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Foo { fn foo() {} }

impl<T> Foo for T {}
impl<U> Foo for U {} //~ ERROR conflicting implementations of trait `Foo`:

trait Bar { fn bar() {} }

impl<T> Bar for (T, u8) {}
impl<T> Bar for (u8, T) {} //~ ERROR conflicting implementations of trait `Bar` for type `(u8, u8)`:

trait Baz<T> { fn baz() {} }

impl<T> Baz<u8> for T {}
impl<T> Baz<T> for u8 {} //~ ERROR conflicting implementations of trait `Baz<u8>` for type `u8`:

trait Quux<U, V> { fn quux() {} }

impl<T, U, V> Quux<U, V> for T {}
impl<T, U> Quux<U, U> for T {} //~ ERROR conflicting implementations of trait `Quux<_, _>`:
impl<T, V> Quux<T, V> for T {} //~ ERROR conflicting implementations of trait `Quux<_, _>`:

fn main() {}
