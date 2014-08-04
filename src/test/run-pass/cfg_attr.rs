// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:--cfg set1 --cfg set2
#![allow(dead_code)]
use std::fmt::Show;

struct NotShowable;

#[cfg_attr(set1, deriving(Show))]
struct Set1;

#[cfg_attr(notset, deriving(Show))]
struct Notset(NotShowable);

#[cfg_attr(not(notset), deriving(Show))]
struct NotNotset;

#[cfg_attr(not(set1), deriving(Show))]
struct NotSet1(NotShowable);

#[cfg_attr(all(set1, set2), deriving(Show))]
struct AllSet1Set2;

#[cfg_attr(all(set1, notset), deriving(Show))]
struct AllSet1Notset(NotShowable);

#[cfg_attr(any(set1, notset), deriving(Show))]
struct AnySet1Notset;

#[cfg_attr(any(notset, notset2), deriving(Show))]
struct AnyNotsetNotset2(NotShowable);

#[cfg_attr(all(not(notset), any(set1, notset)), deriving(Show))]
struct Complex;

#[cfg_attr(any(notset, not(any(set1, notset))), deriving(Show))]
struct ComplexNot(NotShowable);

#[cfg_attr(any(target_endian = "little", target_endian = "big"), deriving(Show))]
struct KeyValue;

fn is_show<T: Show>() {}

fn main() {
    is_show::<Set1>();
    is_show::<NotNotset>();
    is_show::<AllSet1Set2>();
    is_show::<AnySet1Notset>();
    is_show::<Complex>();
    is_show::<KeyValue>();
}
