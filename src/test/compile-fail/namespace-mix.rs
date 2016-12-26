// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:namespace-mix.rs

extern crate namespace_mix;
use namespace_mix::*;

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

// Use something emitting the type argument name, e.g. unsatisfied bound.
trait Impossible {}
fn check<T: Impossible>(_: T) {}

mod m1 {
    pub use ::c::*;
    pub type S = ::c::Item;
}
mod m2 {
    pub use ::c::*;
    pub const S: ::c::Item = ::c::Item;
}

fn f12() {
    check(m1::S{}); //~ ERROR c::Item
    check(m1::S); //~ ERROR expected value, found type alias `m1::S`
    check(m2::S{}); //~ ERROR c::S
    check(m2::S); //~ ERROR c::Item
}
fn xf12() {
    check(xm1::S{}); //~ ERROR c::Item
    check(xm1::S); //~ ERROR expected value, found type alias `xm1::S`
    check(xm2::S{}); //~ ERROR c::S
    check(xm2::S); //~ ERROR c::Item
}

mod m3 {
    pub use ::c::*;
    pub type TS = ::c::Item;
}
mod m4 {
    pub use ::c::*;
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
    pub use ::c::*;
    pub type US = ::c::Item;
}
mod m6 {
    pub use ::c::*;
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
    pub use ::c::E::*;
    pub type V = ::c::Item;
}
mod m8 {
    pub use ::c::E::*;
    pub const V: ::c::Item = ::c::Item;
}

fn f78() {
    check(m7::V{}); //~ ERROR c::Item
    check(m7::V); //~ ERROR expected value, found struct variant `m7::V`
    check(m8::V{}); //~ ERROR c::E
    check(m8::V); //~ ERROR c::Item
}
fn xf78() {
    check(xm7::V{}); //~ ERROR c::Item
    check(xm7::V); //~ ERROR expected value, found struct variant `xm7::V`
    check(xm8::V{}); //~ ERROR c::E
    check(xm8::V); //~ ERROR c::Item
}

mod m9 {
    pub use ::c::E::*;
    pub type TV = ::c::Item;
}
mod mA {
    pub use ::c::E::*;
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
    pub use ::c::E::*;
    pub type UV = ::c::Item;
}
mod mC {
    pub use ::c::E::*;
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
