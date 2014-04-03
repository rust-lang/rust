// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-android: FIXME(#10381)

// compile-flags:-g
// debugger:rbreak zzz
// debugger:run
// debugger:finish
// debugger:print string1.length
// check:$1 = 48
// debugger:print string2.length
// check:$2 = 48
// debugger:print string3.length
// check:$3 = 48
// debugger:continue

#[allow(unused_variable)];

// This test case makes sure that debug info does not ICE when include_str is
// used multiple times (see issue #11322).

fn main() {
    let string1 = include_str!("text-to-include-1.txt");
    let string2 = include_str!("text-to-include-2.txt");
    let string3 = include_str!("text-to-include-3.txt");
    zzz();
}

fn zzz() {()}
