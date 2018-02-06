// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-wasm32-bare compiled with panic=abort by default

#![feature(fn_traits)]
#![feature(never_type)]

use std::panic;

fn foo(x: u32, y: u32) -> u32 { x/y }
fn foo_diverges() -> ! { panic!() }

fn test_fn_ptr<T>(mut t: T)
    where T: Fn(u32, u32) -> u32,
{
    let as_fn = <T as Fn<(u32, u32)>>::call;
    assert_eq!(as_fn(&t, (9, 3)), 3);
    let as_fn_mut = <T as FnMut<(u32, u32)>>::call_mut;
    assert_eq!(as_fn_mut(&mut t, (18, 3)), 6);
    let as_fn_once = <T as FnOnce<(u32, u32)>>::call_once;
    assert_eq!(as_fn_once(t, (24, 3)), 8);
}

fn assert_panics<F>(f: F) where F: FnOnce() {
    let f = panic::AssertUnwindSafe(f);
    let result = panic::catch_unwind(move || {
        f.0()
    });
    if let Ok(..) = result {
        panic!("diverging function returned");
    }
}

fn test_fn_ptr_panic<T>(mut t: T)
    where T: Fn() -> !
{
    let as_fn = <T as Fn<()>>::call;
    assert_panics(|| as_fn(&t, ()));
    let as_fn_mut = <T as FnMut<()>>::call_mut;
    assert_panics(|| as_fn_mut(&mut t, ()));
    let as_fn_once = <T as FnOnce<()>>::call_once;
    assert_panics(|| as_fn_once(t, ()));
}

fn main() {
    test_fn_ptr(foo);
    test_fn_ptr(foo as fn(u32, u32) -> u32);
    test_fn_ptr_panic(foo_diverges);
    test_fn_ptr_panic(foo_diverges as fn() -> !);
}
