// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// LLDB can't handle zero-sized values
// ignore-lldb

// ignore-android: FIXME(#10381)

// compile-flags:-g
// gdb-command:rbreak zzz
// gdb-command:run
// gdb-command:finish

// gdb-command:print first
// gdb-check:$1 = {<No data fields>}

// gdb-command:print second
// gdb-check:$2 = {<No data fields>}

#![allow(unused_variable)]

enum ANilEnum {}
enum AnotherNilEnum {}

// I (mw) am not sure this test case makes much sense...
// Also, it relies on some implementation details:
// 1. That empty enums as well as '()' are represented as empty structs
// 2. That gdb prints the string "{<No data fields>}" for empty structs (which may change some time)
fn main() {
    unsafe {
        let first: ANilEnum = std::mem::transmute(());
        let second: AnotherNilEnum = std::mem::transmute(());

        zzz();
    }
}

fn zzz() {()}
