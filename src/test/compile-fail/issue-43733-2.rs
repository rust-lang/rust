// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(const_fn)]
#![feature(cfg_target_thread_local, thread_local_internals)]

// On platforms *without* `#[thread_local]`, use
// a custom non-`Sync` type to fake the same error.
#[cfg(not(target_thread_local))]
struct Key<T> {
    _data: std::cell::UnsafeCell<Option<T>>,
    _flag: std::cell::Cell<bool>,
}

#[cfg(not(target_thread_local))]
impl<T> Key<T> {
    const fn new() -> Self {
        Key {
            _data: std::cell::UnsafeCell::new(None),
            _flag: std::cell::Cell::new(false),
        }
    }
}

#[cfg(target_thread_local)]
use std::thread::__FastLocalKeyInner as Key;

static __KEY: Key<()> = Key::new();
//~^ ERROR `std::cell::UnsafeCell<std::option::Option<()>>` cannot be shared between threads
//~| ERROR `std::cell::Cell<bool>` cannot be shared between threads safely [E0277]

fn main() {}
