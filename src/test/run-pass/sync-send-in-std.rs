// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-cloudabi networking not available
// ignore-wasm32-bare networking not available

#![feature(lookup_host)]

use std::net::lookup_host;

fn is_sync<T>(_: T) where T: Sync {}
fn is_send<T>(_: T) where T: Send {}

macro_rules! all_sync_send {
    ($ctor:expr, $($iter:ident),+) => ({
        $(
            let mut x = $ctor;
            is_sync(x.$iter());
            let mut y = $ctor;
            is_send(y.$iter());
        )+
    })
}

fn main() {
    all_sync_send!(lookup_host("localhost").unwrap(), next);
}
