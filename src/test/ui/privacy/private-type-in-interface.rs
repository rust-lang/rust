// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:private-inferred-type.rs

#![allow(warnings)]

extern crate private_inferred_type as ext;

mod m {
    struct Priv;
    pub type Alias = Priv;

    pub trait Trait { type X; }
    impl Trait for Priv { type X = u8; }
}

fn f(_: m::Alias) {} //~ ERROR type `m::Priv` is private
                     //~^ ERROR type `m::Priv` is private
fn f_ext(_: ext::Alias) {} //~ ERROR type `ext::Priv` is private
                           //~^ ERROR type `ext::Priv` is private

trait Tr1 {}
impl m::Alias {} //~ ERROR type `m::Priv` is private
impl Tr1 for ext::Alias {} //~ ERROR type `ext::Priv` is private
type A = <m::Alias as m::Trait>::X; //~ ERROR type `m::Priv` is private

trait Tr2<T> {}
impl<T> Tr2<T> for u8 {}
fn g() -> impl Tr2<m::Alias> { 0 } //~ ERROR type `m::Priv` is private
//~^ ERROR type `m::Priv` is private
fn g_ext() -> impl Tr2<ext::Alias> { 0 } //~ ERROR type `ext::Priv` is private
//~^ ERROR type `ext::Priv` is private

fn main() {}
