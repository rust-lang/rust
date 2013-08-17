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

// debugger:print no_padding16
// check:$1 = {x = 10000, y = -10001}

// debugger:print no_padding32
// check:$2 = {x = -10002, y = -10003.5, z = 10004}

// debugger:print no_padding64
// check:$3 = {x = -10005.5, y = 10006, z = 10007}

// debugger:print no_padding163264
// check:$4 = {a = -10008, b = 10009, c = 10010, d = 10011}

// debugger:print internal_padding
// check:$5 = {x = 10012, y = -10013}

// debugger:print padding_at_end
// check:$6 = {x = -10014, y = 10015}

#[allow(unused_variable)];

struct NoPadding16 {
    x: u16,
    y: i16
}

struct NoPadding32 {
    x: i32,
    y: f32,
    z: u32
}

struct NoPadding64 {
    x: f64,
    y: i64,
    z: u64
}

struct NoPadding163264 {
    a: i16,
    b: u16,
    c: i32,
    d: u64
}

struct InternalPadding {
    x: u16,
    y: i64
}

struct PaddingAtEnd {
    x: i64,
    y: u16
}

fn main() {
    let no_padding16 = NoPadding16 { x: 10000, y: -10001 };
    let no_padding32 = NoPadding32 { x: -10002, y: -10003.5, z: 10004 };
    let no_padding64 = NoPadding64 { x: -10005.5, y: 10006, z: 10007 };
    let no_padding163264 = NoPadding163264 { a: -10008, b: 10009, c: 10010, d: 10011 };

    let internal_padding = InternalPadding { x: 10012, y: -10013 };
    let padding_at_end = PaddingAtEnd { x: -10014, y: 10015 };

    zzz();
}

fn zzz() {()}
