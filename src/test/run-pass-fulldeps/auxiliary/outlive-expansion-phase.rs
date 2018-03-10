// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// force-host

#![feature(plugin_registrar)]
#![feature(box_syntax, rustc_private)]

extern crate rustc;
extern crate rustc_plugin;

use std::any::Any;
use std::cell::RefCell;
use rustc_plugin::Registry;

struct Foo {
    foo: isize
}

impl Drop for Foo {
    fn drop(&mut self) {}
}

#[plugin_registrar]
pub fn registrar(_: &mut Registry) {
    thread_local!(static FOO: RefCell<Option<Box<Any+Send>>> = RefCell::new(None));
    FOO.with(|s| *s.borrow_mut() = Some(box Foo { foo: 10 } as Box<Any+Send>));
}
