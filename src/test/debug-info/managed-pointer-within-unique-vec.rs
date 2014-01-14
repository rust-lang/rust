// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-android: FIXME(#10381)

#[feature(managed_boxes)];

// compile-flags:-Z extra-debug-info
// debugger:rbreak zzz
// debugger:run
// debugger:finish

// debugger:print unique->elements[0]->val
// check:$1 = 10

// debugger:print unique->elements[1]->val
// check:$2 = 11

// debugger:print unique->elements[2]->val
// check:$3 = 12

// debugger:print unique->elements[3]->val
// check:$4 = 13

#[allow(unused_variable)];

fn main() {

    let unique: ~[@i64] = ~[@10, @11, @12, @13];

    zzz();
}

fn zzz() {()}
