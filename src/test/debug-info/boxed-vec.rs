// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:-Z extra-debug-info
// debugger:rbreak zzz
// debugger:run
// debugger:finish

// debugger:print managed->val.fill
// check:$1 = 24
// debugger:print *((uint64_t[3]*)(managed->val.elements))
// check:$2 = {7, 8, 9}

// debugger:print unique->fill
// check:$3 = 32
// debugger:print *((uint64_t[4]*)(unique->elements))
// check:$4 = {10, 11, 12, 13}

#[allow(unused_variable)];

fn main() {

    let managed: @[i64] = @[7, 8, 9];
    let unique: ~[i64] = ~[10, 11, 12, 13];

    zzz();
}

fn zzz() {()}
