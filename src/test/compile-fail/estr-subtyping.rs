// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(managed_boxes)];

fn wants_box(x: @str) { }
fn wants_uniq(x: ~str) { }
fn wants_slice(x: &str) { }

fn has_box(x: @str) {
   wants_box(x);
   wants_uniq(x); //~ ERROR str storage differs: expected `~` but found `@`
   wants_slice(x);
}

fn has_uniq(x: ~str) {
   wants_box(x); //~ ERROR str storage differs: expected `@` but found `~`
   wants_uniq(x);
   wants_slice(x);
}

fn has_slice(x: &str) {
   wants_box(x); //~ ERROR str storage differs: expected `@` but found `&`
   wants_uniq(x); //~ ERROR str storage differs: expected `~` but found `&`
   wants_slice(x);
}

fn main() {
}
