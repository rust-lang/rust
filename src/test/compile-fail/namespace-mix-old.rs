// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME: Remove when `item_like_imports` is stabilized.

// aux-build:namespace-mix-old.rs

#![feature(relaxed_adts)]

extern crate namespace_mix_old;
use namespace_mix_old::{xm1, xm2, xm3, xm4, xm5, xm6, xm7, xm8, xm9, xmA, xmB, xmC};

mod c {
    pub struct S {}
    pub struct TS();
    pub struct US;
    pub enum E {
        V {},
        TV(),
        UV,
    }

    pub struct Item;
}

mod proxy {
    pub use c::*;
    pub use c::E::*;
}

// Use something emitting the type argument name, e.g. unsatisfied bound.
trait Impossible {}
fn check<T: Impossible>(_: T) {}

mod m1 {
    pub use ::proxy::*;
    pub type S = ::c::Item;
}
mod m2 {
    pub use ::proxy::*;
    pub const S: ::c::Item = ::c::Item;
}

fn f12() {
    check(m1::S{}); //~ ERROR c::Item
    check(m1::S); //~ ERROR unresolved name
    check(m2::S{}); //~ ERROR c::S
    check(m2::S); //~ ERROR c::Item
}
fn xf12() {
    check(xm1::S{}); //~ ERROR c::Item
    check(xm1::S); //~ ERROR unresolved name
    check(xm2::S{}); //~ ERROR c::S
    check(xm2::S); //~ ERROR c::Item
}

mod m3 {
    pub use ::proxy::*;
    pub type TS = ::c::Item;
}
mod m4 {
    pub use ::proxy::*;
    pub const TS: ::c::Item = ::c::Item;
}

fn f34() {
    check(m3::TS{}); //~ ERROR c::Item
    check(m3::TS); //~ ERROR c::TS
    check(m4::TS{}); //~ ERROR c::TS
    check(m4::TS); //~ ERROR c::Item
}
fn xf34() {
    check(xm3::TS{}); //~ ERROR c::Item
    check(xm3::TS); //~ ERROR c::TS
    check(xm4::TS{}); //~ ERROR c::TS
    check(xm4::TS); //~ ERROR c::Item
}

mod m5 {
    pub use ::proxy::*;
    pub type US = ::c::Item;
}
mod m6 {
    pub use ::proxy::*;
    pub const US: ::c::Item = ::c::Item;
}

fn f56() {
    check(m5::US{}); //~ ERROR c::Item
    check(m5::US); //~ ERROR c::US
    check(m6::US{}); //~ ERROR c::US
    check(m6::US); //~ ERROR c::Item
}
fn xf56() {
    check(xm5::US{}); //~ ERROR c::Item
    check(xm5::US); //~ ERROR c::US
    check(xm6::US{}); //~ ERROR c::US
    check(xm6::US); //~ ERROR c::Item
}

mod m7 {
    pub use ::proxy::*;
    pub type V = ::c::Item;
}
mod m8 {
    pub use ::proxy::*;
    pub const V: ::c::Item = ::c::Item;
}

fn f78() {
    check(m7::V{}); //~ ERROR c::Item
    check(m7::V); //~ ERROR name of a struct or struct variant
    check(m8::V{}); //~ ERROR c::E
    check(m8::V); //~ ERROR c::Item
}
fn xf78() {
    check(xm7::V{}); //~ ERROR c::Item
    check(xm7::V); //~ ERROR name of a struct or struct variant
    check(xm8::V{}); //~ ERROR c::E
    check(xm8::V); //~ ERROR c::Item
}

mod m9 {
    pub use ::proxy::*;
    pub type TV = ::c::Item;
}
mod mA {
    pub use ::proxy::*;
    pub const TV: ::c::Item = ::c::Item;
}

fn f9A() {
    check(m9::TV{}); //~ ERROR c::Item
    check(m9::TV); //~ ERROR c::E
    check(mA::TV{}); //~ ERROR c::E
    check(mA::TV); //~ ERROR c::Item
}
fn xf9A() {
    check(xm9::TV{}); //~ ERROR c::Item
    check(xm9::TV); //~ ERROR c::E
    check(xmA::TV{}); //~ ERROR c::E
    check(xmA::TV); //~ ERROR c::Item
}

mod mB {
    pub use ::proxy::*;
    pub type UV = ::c::Item;
}
mod mC {
    pub use ::proxy::*;
    pub const UV: ::c::Item = ::c::Item;
}

fn fBC() {
    check(mB::UV{}); //~ ERROR c::Item
    check(mB::UV); //~ ERROR c::E
    check(mC::UV{}); //~ ERROR c::E
    check(mC::UV); //~ ERROR c::Item
}
fn xfBC() {
    check(xmB::UV{}); //~ ERROR c::Item
    check(xmB::UV); //~ ERROR c::E
    check(xmC::UV{}); //~ ERROR c::E
    check(xmC::UV); //~ ERROR c::Item
}

fn main() {}
