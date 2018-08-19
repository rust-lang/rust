// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// check that moves due to a closure capture give a special note

#![feature(nll)]

fn move_after_move(x: String) {
    || x;
    let y = x; //~ ERROR
}

fn borrow_after_move(x: String) {
    || x;
    let y = &x; //~ ERROR
}

fn borrow_mut_after_move(mut x: String) {
    || x;
    let y = &mut x; //~ ERROR
}

fn fn_ref<F: Fn()>(f: F) -> F { f }
fn fn_mut<F: FnMut()>(f: F) -> F { f }

fn main() {}
