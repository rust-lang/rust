// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-win32 Broken because of LLVM bug: http://llvm.org/bugs/show_bug.cgi?id=16249

// compile-flags:-Z extra-debug-info
// debugger:set print pretty off
// debugger:break zzz
// debugger:run
// debugger:finish

// debugger:print/d noPadding8
// check:$1 = {-100, 100}
// debugger:print noPadding16
// check:$2 = {0, 1, 2}
// debugger:print noPadding32
// check:$3 = {3, 4.5, 5}
// debugger:print noPadding64
// check:$4 = {6, 7.5, 8}

// debugger:print internalPadding1
// check:$5 = {9, 10}
// debugger:print internalPadding2
// check:$6 = {11, 12, 13, 14}

// debugger:print paddingAtEnd
// check:$7 = {15, 16}


fn main() {
    let noPadding8: (i8, u8) = (-100, 100);
    let noPadding16: (i16, i16, u16) = (0, 1, 2);
    let noPadding32: (i32, f32, u32) = (3, 4.5, 5);
    let noPadding64: (i64, f64, u64) = (6, 7.5, 8);

    let internalPadding1: (i16, i32) = (9, 10);
    let internalPadding2: (i16, i32, u32, u64) = (11, 12, 13, 14);

    let paddingAtEnd: (i32, i16) = (15, 16);

    zzz();
}

fn zzz() {()}