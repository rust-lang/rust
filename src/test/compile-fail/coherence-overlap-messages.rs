// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Foo {}

impl<T> Foo for T {} //~ ERROR conflicting implementations of trait `Foo`:
impl<U> Foo for U {}

trait Bar {}

impl<T> Bar for T {} //~ ERROR conflicting implementations of trait `Bar` for type `u8`:
impl Bar for u8 {}

trait Baz<T> {}

impl<T, U> Baz<U> for T {} //~ ERROR conflicting implementations of trait `Baz<_>` for type `u8`:
impl<T> Baz<T> for u8 {}

trait Quux<T> {}

impl<T, U> Quux<U> for T {} //~ ERROR conflicting implementations of trait `Quux<_>`:
impl<T> Quux<T> for T {}

trait Qaar<T> {}

impl<T, U> Qaar<U> for T {} //~ ERROR conflicting implementations of trait `Qaar<u8>`:
impl<T> Qaar<u8> for T {}

trait Qaax<T> {}

impl<T, U> Qaax<U> for T {}
//~^ ERROR conflicting implementations of trait `Qaax<u8>` for type `u32`:
impl Qaax<u8> for u32 {}

fn main() {}
