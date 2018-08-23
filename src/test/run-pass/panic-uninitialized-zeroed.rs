// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test checks that instantiating an uninhabited type via `mem::{uninitialized,zeroed}` results
// in a runtime panic.

#![feature(never_type)]

use std::{mem, panic};

struct Foo {
    x: u8,
    y: !,
}

enum Bar {}

fn main() {
    unsafe {
        assert_eq!(
            panic::catch_unwind(|| {
                mem::uninitialized::<!>()
            }).err().and_then(|a| a.downcast_ref::<String>().map(|s| {
                s == "Attempted to instantiate uninhabited type ! using mem::uninitialized"
            })),
            Some(true)
        );

        assert_eq!(
            panic::catch_unwind(|| {
                mem::zeroed::<!>()
            }).err().and_then(|a| a.downcast_ref::<String>().map(|s| {
                s == "Attempted to instantiate uninhabited type ! using mem::zeroed"
            })),
            Some(true)
        );

        assert_eq!(
            panic::catch_unwind(|| {
                mem::uninitialized::<Foo>()
            }).err().and_then(|a| a.downcast_ref::<String>().map(|s| {
                s == "Attempted to instantiate uninhabited type Foo using mem::uninitialized"
            })),
            Some(true)
        );

        assert_eq!(
            panic::catch_unwind(|| {
                mem::zeroed::<Foo>()
            }).err().and_then(|a| a.downcast_ref::<String>().map(|s| {
                s == "Attempted to instantiate uninhabited type Foo using mem::zeroed"
            })),
            Some(true)
        );

        assert_eq!(
            panic::catch_unwind(|| {
                mem::uninitialized::<Bar>()
            }).err().and_then(|a| a.downcast_ref::<String>().map(|s| {
                s == "Attempted to instantiate uninhabited type Bar using mem::uninitialized"
            })),
            Some(true)
        );

        assert_eq!(
            panic::catch_unwind(|| {
                mem::zeroed::<Bar>()
            }).err().and_then(|a| a.downcast_ref::<String>().map(|s| {
                s == "Attempted to instantiate uninhabited type Bar using mem::zeroed"
            })),
            Some(true)
        );
    }
}
