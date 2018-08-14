// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

struct S {f:String}
impl Drop for S {
    fn drop(&mut self) { println!("{}", self.f); }
}

fn move_in_match() {
    match (S {f:"foo".to_string()}) {
        //[mir]~^ ERROR [E0509]
        S {f:_s} => {}
        //[ast]~^ ERROR cannot move out of type `S`, which implements the `Drop` trait [E0509]
    }
}

fn move_in_let() {
    let S {f:_s} = S {f:"foo".to_string()};
    //[ast]~^ ERROR cannot move out of type `S`, which implements the `Drop` trait [E0509]
    //[mir]~^^ ERROR [E0509]
}

fn move_in_fn_arg(S {f:_s}: S) {
    //[ast]~^ ERROR cannot move out of type `S`, which implements the `Drop` trait [E0509]
    //[mir]~^^ ERROR [E0509]
}

fn main() {}
