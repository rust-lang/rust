// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(core_intrinsics)]
#![allow(warnings)]

use std::intrinsics;

#[derive(Copy, Clone)]
struct Foo(i64);
type Bar = &'static Fn();
type Quux = [u8; 100];

unsafe fn test_bool_load(p: &mut bool, v: bool) {
    intrinsics::atomic_load(p);
    //~^ ERROR `atomic_load` intrinsic: expected basic integer type, found `bool`
}

unsafe fn test_bool_store(p: &mut bool, v: bool) {
    intrinsics::atomic_store(p, v);
    //~^ ERROR `atomic_store` intrinsic: expected basic integer type, found `bool`
}

unsafe fn test_bool_xchg(p: &mut bool, v: bool) {
    intrinsics::atomic_xchg(p, v);
    //~^ ERROR `atomic_xchg` intrinsic: expected basic integer type, found `bool`
}

unsafe fn test_bool_cxchg(p: &mut bool, v: bool) {
    intrinsics::atomic_cxchg(p, v, v);
    //~^ ERROR `atomic_cxchg` intrinsic: expected basic integer type, found `bool`
}

unsafe fn test_Foo_load(p: &mut Foo, v: Foo) {
    intrinsics::atomic_load(p);
    //~^ ERROR `atomic_load` intrinsic: expected basic integer type, found `Foo`
}

unsafe fn test_Foo_store(p: &mut Foo, v: Foo) {
    intrinsics::atomic_store(p, v);
    //~^ ERROR `atomic_store` intrinsic: expected basic integer type, found `Foo`
}

unsafe fn test_Foo_xchg(p: &mut Foo, v: Foo) {
    intrinsics::atomic_xchg(p, v);
    //~^ ERROR `atomic_xchg` intrinsic: expected basic integer type, found `Foo`
}

unsafe fn test_Foo_cxchg(p: &mut Foo, v: Foo) {
    intrinsics::atomic_cxchg(p, v, v);
    //~^ ERROR `atomic_cxchg` intrinsic: expected basic integer type, found `Foo`
}

unsafe fn test_Bar_load(p: &mut Bar, v: Bar) {
    intrinsics::atomic_load(p);
    //~^ ERROR expected basic integer type, found `&std::ops::Fn()`
}

unsafe fn test_Bar_store(p: &mut Bar, v: Bar) {
    intrinsics::atomic_store(p, v);
    //~^ ERROR expected basic integer type, found `&std::ops::Fn()`
}

unsafe fn test_Bar_xchg(p: &mut Bar, v: Bar) {
    intrinsics::atomic_xchg(p, v);
    //~^ ERROR expected basic integer type, found `&std::ops::Fn()`
}

unsafe fn test_Bar_cxchg(p: &mut Bar, v: Bar) {
    intrinsics::atomic_cxchg(p, v, v);
    //~^ ERROR expected basic integer type, found `&std::ops::Fn()`
}

unsafe fn test_Quux_load(p: &mut Quux, v: Quux) {
    intrinsics::atomic_load(p);
    //~^ ERROR `atomic_load` intrinsic: expected basic integer type, found `[u8; 100]`
}

unsafe fn test_Quux_store(p: &mut Quux, v: Quux) {
    intrinsics::atomic_store(p, v);
    //~^ ERROR `atomic_store` intrinsic: expected basic integer type, found `[u8; 100]`
}

unsafe fn test_Quux_xchg(p: &mut Quux, v: Quux) {
    intrinsics::atomic_xchg(p, v);
    //~^ ERROR `atomic_xchg` intrinsic: expected basic integer type, found `[u8; 100]`
}

unsafe fn test_Quux_cxchg(p: &mut Quux, v: Quux) {
    intrinsics::atomic_cxchg(p, v, v);
    //~^ ERROR `atomic_cxchg` intrinsic: expected basic integer type, found `[u8; 100]`
}

fn main() {}
