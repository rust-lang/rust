// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(const_fn, drop_types_in_const)]
#![feature(cfg_target_thread_local, thread_local_internals, thread_local_state)]

type Foo = std::cell::RefCell<String>;

static __KEY: std::thread::__FastLocalKeyInner<Foo> =
    std::thread::__FastLocalKeyInner::new();

#[cfg(not(target_thread_local))]
static __KEY: std::thread::__OsLocalKeyInner<Foo> =
    std::thread::__OsLocalKeyInner::new();

fn __getit() -> &'static std::option::Option<Foo> {
    __KEY.get() //~ ERROR  invocation of unsafe method requires unsafe
}

fn __get_state() -> std::thread::LocalKeyState {
    __KEY.get_state() //~ ERROR  invocation of unsafe method requires unsafe
}

fn __pre_init() {
    __KEY.pre_init() //~ ERROR  invocation of unsafe method requires unsafe
}

fn __post_init(val: Foo) {
    __KEY.post_init(val) //~ ERROR  invocation of unsafe method requires unsafe
}

fn __rollback_init() {
    __KEY.rollback_init() //~ ERROR  invocation of unsafe method requires unsafe
}

static FOO: std::thread::LocalKey<Foo> =
    std::thread::LocalKey::new(__getit, //~ ERROR call to unsafe function requires unsafe
                               __get_state,
                               __pre_init,
                               Default::default,
                               __post_init,
                               __rollback_init);

fn main() {
    FOO.with(|foo| println!("{}", foo.borrow()));
    std::thread::spawn(|| {
        FOO.with(|foo| *foo.borrow_mut() += "foo");
    }).join().unwrap();
    FOO.with(|foo| println!("{}", foo.borrow()));
}
