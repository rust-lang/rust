// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-win32: FIXME #13256
// ignore-android: FIXME(#10381)

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:set print pretty off
// gdb-command:rbreak zzz

// gdb-command:print 'simple-struct::NO_PADDING_16'
// gdb-check:$1 = {x = 1000, y = -1001}

// gdb-command:print 'simple-struct::NO_PADDING_32'
// gdb-check:$2 = {x = 1, y = 2, z = 3}

// gdb-command:print 'simple-struct::NO_PADDING_64'
// gdb-check:$3 = {x = 4, y = 5, z = 6}

// gdb-command:print 'simple-struct::NO_PADDING_163264'
// gdb-check:$4 = {a = 7, b = 8, c = 9, d = 10}

// gdb-command:print 'simple-struct::INTERNAL_PADDING'
// gdb-check:$5 = {x = 11, y = 12}

// gdb-command:print 'simple-struct::PADDING_AT_END'
// gdb-check:$6 = {x = 13, y = 14}

// gdb-command:run
// gdb-command:finish

// gdb-command:print no_padding16
// gdb-check:$7 = {x = 10000, y = -10001}

// gdb-command:print no_padding32
// gdb-check:$8 = {x = -10002, y = -10003.5, z = 10004}

// gdb-command:print no_padding64
// gdb-check:$9 = {x = -10005.5, y = 10006, z = 10007}

// gdb-command:print no_padding163264
// gdb-check:$10 = {a = -10008, b = 10009, c = 10010, d = 10011}

// gdb-command:print internal_padding
// gdb-check:$11 = {x = 10012, y = -10013}

// gdb-command:print padding_at_end
// gdb-check:$12 = {x = -10014, y = 10015}

// gdb-command:print 'simple-struct::NO_PADDING_16'
// gdb-check:$13 = {x = 100, y = -101}

// gdb-command:print 'simple-struct::NO_PADDING_32'
// gdb-check:$14 = {x = -15, y = -16, z = 17}

// gdb-command:print 'simple-struct::NO_PADDING_64'
// gdb-check:$15 = {x = -18, y = 19, z = 20}

// gdb-command:print 'simple-struct::NO_PADDING_163264'
// gdb-check:$16 = {a = -21, b = 22, c = 23, d = 24}

// gdb-command:print 'simple-struct::INTERNAL_PADDING'
// gdb-check:$17 = {x = 25, y = -26}

// gdb-command:print 'simple-struct::PADDING_AT_END'
// gdb-check:$18 = {x = -27, y = 28}

// gdb-command:print inheriting
// gdb-check:$19 = {a = 10019, b = -10020, x = -10016, y = -10017.5, z = 10018}


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print no_padding16
// lldb-check:[...]$0 = NoPadding16 { x: 10000, y: -10001 }

// lldb-command:print no_padding32
// lldb-check:[...]$1 = NoPadding32 { x: -10002, y: -10003.5, z: 10004 }

// lldb-command:print no_padding64
// lldb-check:[...]$2 = NoPadding64 { x: -10005.5, y: 10006, z: 10007 }

// lldb-command:print no_padding163264
// lldb-check:[...]$3 = NoPadding163264 { a: -10008, b: 10009, c: 10010, d: 10011 }

// lldb-command:print internal_padding
// lldb-check:[...]$4 = InternalPadding { x: 10012, y: -10013 }

// lldb-command:print padding_at_end
// lldb-check:[...]$5 = PaddingAtEnd { x: -10014, y: 10015 }

#![feature(struct_inherit)];
#![allow(unused_variable)];
#![allow(dead_code)];

struct NoPadding16 {
    x: u16,
    y: i16
}

virtual struct NoPadding32 {
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

static mut NO_PADDING_16: NoPadding16 = NoPadding16 {
    x: 1000,
    y: -1001
};

static mut NO_PADDING_32: NoPadding32 = NoPadding32 {
    x: 1,
    y: 2.0,
    z: 3
};

static mut NO_PADDING_64: NoPadding64 = NoPadding64 {
    x: 4.0,
    y: 5,
    z: 6
};

static mut NO_PADDING_163264: NoPadding163264 = NoPadding163264 {
    a: 7,
    b: 8,
    c: 9,
    d: 10
};

static mut INTERNAL_PADDING: InternalPadding = InternalPadding {
    x: 11,
    y: 12
};

static mut PADDING_AT_END: PaddingAtEnd = PaddingAtEnd {
    x: 13,
    y: 14
};

struct Inheriting : NoPadding32 {
    a: u16,
    b: i16
}

fn main() {
    let no_padding16 = NoPadding16 { x: 10000, y: -10001 };
    let no_padding32 = NoPadding32 { x: -10002, y: -10003.5, z: 10004 };
    let no_padding64 = NoPadding64 { x: -10005.5, y: 10006, z: 10007 };
    let no_padding163264 = NoPadding163264 { a: -10008, b: 10009, c: 10010, d: 10011 };

    let internal_padding = InternalPadding { x: 10012, y: -10013 };
    let padding_at_end = PaddingAtEnd { x: -10014, y: 10015 };

    let inheriting = Inheriting { a: 10019, b: -10020, x: -10016, y: -10017.5, z: 10018 };

    unsafe {
        NO_PADDING_16.x = 100;
        NO_PADDING_16.y = -101;

        NO_PADDING_32.x = -15;
        NO_PADDING_32.y = -16.0;
        NO_PADDING_32.z = 17;

        NO_PADDING_64.x = -18.0;
        NO_PADDING_64.y = 19;
        NO_PADDING_64.z = 20;

        NO_PADDING_163264.a = -21;
        NO_PADDING_163264.b = 22;
        NO_PADDING_163264.c = 23;
        NO_PADDING_163264.d = 24;

        INTERNAL_PADDING.x = 25;
        INTERNAL_PADDING.y = -26;

        PADDING_AT_END.x = -27;
        PADDING_AT_END.y = 28;
    }

    zzz(); // #break
}

fn zzz() {()}
