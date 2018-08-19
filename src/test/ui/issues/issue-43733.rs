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
#![feature(thread_local)]
#![feature(cfg_target_thread_local, thread_local_internals)]

type Foo = std::cell::RefCell<String>;

#[cfg(target_thread_local)]
#[thread_local]
static __KEY: std::thread::__FastLocalKeyInner<Foo> =
    std::thread::__FastLocalKeyInner::new();

#[cfg(not(target_thread_local))]
static __KEY: std::thread::__OsLocalKeyInner<Foo> =
    std::thread::__OsLocalKeyInner::new();

fn __getit() -> std::option::Option<
    &'static std::cell::UnsafeCell<
        std::option::Option<Foo>>>
{
    __KEY.get() //~ ERROR call to unsafe function is unsafe
}

static FOO: std::thread::LocalKey<Foo> =
    std::thread::LocalKey::new(__getit, Default::default);
//~^ ERROR call to unsafe function is unsafe

fn main() {
    FOO.with(|foo| println!("{}", foo.borrow()));
    std::thread::spawn(|| {
        FOO.with(|foo| *foo.borrow_mut() += "foo");
    }).join().unwrap();
    FOO.with(|foo| println!("{}", foo.borrow()));
}
