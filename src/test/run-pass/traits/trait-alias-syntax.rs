// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(trait_alias)]

trait SimpleAlias = Default;
trait GenericAlias<T> = Iterator<Item = T>;
trait Partial<T> = IntoIterator<Item = T>;
trait SpecificAlias = GenericAlias<i32>;
trait PartialEqRef<'a, T: 'a> = PartialEq<&'a T>;
trait StaticAlias = 'static;

trait Things<T> {}
trait Romeo {}
#[allow(dead_code)]
struct The<T>(T);
#[allow(dead_code)]
struct Fore<T>(T);
impl<T, U> Things<T> for The<U> {}
impl<T> Romeo for Fore<T> {}

trait WithWhere<Art, Thou> = Romeo + Romeo where Fore<(Art, Thou)>: Romeo;
trait BareWhere<Wild, Are> = where The<Wild>: Things<Are>;

fn main() {}
