// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct S(String);
impl Drop for S {
    fn drop(&mut self) { }
}

fn move_in_match() {
    match S("foo".to_string()) {
        S(_s) => {}
        //~^ ERROR cannot move out of type `S`, which defines the `Drop` trait
    }
}

fn move_in_let() {
    let S(_s) = S("foo".to_string());
    //~^ ERROR cannot move out of type `S`, which defines the `Drop` trait
}

fn move_in_fn_arg(S(_s): S) {
    //~^ ERROR cannot move out of type `S`, which defines the `Drop` trait
}

fn main() {}
