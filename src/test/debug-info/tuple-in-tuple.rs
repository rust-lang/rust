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

// debugger:print noPadding1
// check:$1 = {{0, 1}, 2, 3}
// debugger:print noPadding2
// check:$2 = {4, {5, 6}, 7}
// debugger:print noPadding3
// check:$3 = {8, 9, {10, 11}}

// debugger:print internalPadding1
// check:$4 = {12, {13, 14}}
// debugger:print internalPadding2
// check:$5 = {15, {16, 17}}

// debugger:print paddingAtEnd1
// check:$6 = {18, {19, 20}}
// debugger:print paddingAtEnd2
// check:$7 = {{21, 22}, 23}

fn main() {
    let noPadding1 : ((u32, u32), u32, u32) = ((0, 1), 2, 3);
    let noPadding2 : (u32, (u32, u32), u32) = (4, (5, 6), 7);
    let noPadding3 : (u32, u32, (u32, u32)) = (8, 9, (10, 11));

    let internalPadding1 : (i16, (i32, i32)) = (12, (13, 14));
    let internalPadding2 : (i16, (i16, i32)) = (15, (16, 17));

    let paddingAtEnd1 : (i32, (i32, i16)) = (18, (19, 20));
    let paddingAtEnd2 : ((i32, i16), i32) = ((21, 22), 23);

    zzz();
}

fn zzz() {()}