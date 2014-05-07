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
// gdb-command:set print pretty off
// gdb-command:rbreak zzz
// gdb-command:run
// gdb-command:finish

// gdb-command:print no_padding16
// gdb-check:$1 = {10000, -10001}

// gdb-command:print no_padding32
// gdb-check:$2 = {-10002, -10003.5, 10004}

// gdb-command:print no_padding64
// gdb-check:$3 = {-10005.5, 10006, 10007}

// gdb-command:print no_padding163264
// gdb-check:$4 = {-10008, 10009, 10010, 10011}

// gdb-command:print internal_padding
// gdb-check:$5 = {10012, -10013}

// gdb-command:print padding_at_end
// gdb-check:$6 = {-10014, 10015}


// This test case mainly makes sure that no field names are generated for tuple structs (as opposed
// to all fields having the name "<unnamed_field>"). Otherwise they are handled the same a normal
// structs.

struct NoPadding16(u16, i16);
struct NoPadding32(i32, f32, u32);
struct NoPadding64(f64, i64, u64);
struct NoPadding163264(i16, u16, i32, u64);
struct InternalPadding(u16, i64);
struct PaddingAtEnd(i64, u16);

fn main() {
    let no_padding16 = NoPadding16(10000, -10001);
    let no_padding32 = NoPadding32(-10002, -10003.5, 10004);
    let no_padding64 = NoPadding64(-10005.5, 10006, 10007);
    let no_padding163264 = NoPadding163264(-10008, 10009, 10010, 10011);

    let internal_padding = InternalPadding(10012, -10013);
    let padding_at_end = PaddingAtEnd(-10014, 10015);

    zzz();
}

fn zzz() {()}
