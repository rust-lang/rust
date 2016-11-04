// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength

// min-lldb-version: 310

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print *t0
// gdb-check:$1 = 1
// gdb-command:print *t1
// gdb-check:$2 = 2.5
// gdb-command:print ret
// gdbg-check:$3 = {__0 = {__0 = 1, __1 = 2.5}, __1 = {__0 = 2.5, __1 = 1}}
// gdbr-check:$3 = ((1, 2.5), (2.5, 1))
// gdb-command:continue

// gdb-command:print *t0
// gdb-check:$4 = 3.5
// gdb-command:print *t1
// gdb-check:$5 = 4
// gdb-command:print ret
// gdbg-check:$6 = {__0 = {__0 = 3.5, __1 = 4}, __1 = {__0 = 4, __1 = 3.5}}
// gdbr-check:$6 = ((3.5, 4), (4, 3.5))
// gdb-command:continue

// gdb-command:print *t0
// gdb-check:$7 = 5
// gdb-command:print *t1
// gdbg-check:$8 = {a = 6, b = 7.5}
// gdbr-check:$8 = generic_function::Struct {a: 6, b: 7.5}
// gdb-command:print ret
// gdbg-check:$9 = {__0 = {__0 = 5, __1 = {a = 6, b = 7.5}}, __1 = {__0 = {a = 6, b = 7.5}, __1 = 5}}
// gdbr-check:$9 = ((5, generic_function::Struct {a: 6, b: 7.5}), (generic_function::Struct {a: 6, b: 7.5}, 5))
// gdb-command:continue


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print *t0
// lldb-check:[...]$0 = 1
// lldb-command:print *t1
// lldb-check:[...]$1 = 2.5
// lldb-command:print ret
// lldb-check:[...]$2 = ((1, 2.5), (2.5, 1))
// lldb-command:continue

// lldb-command:print *t0
// lldb-check:[...]$3 = 3.5
// lldb-command:print *t1
// lldb-check:[...]$4 = 4
// lldb-command:print ret
// lldb-check:[...]$5 = ((3.5, 4), (4, 3.5))
// lldb-command:continue

// lldb-command:print *t0
// lldb-check:[...]$6 = 5
// lldb-command:print *t1
// lldb-check:[...]$7 = Struct { a: 6, b: 7.5 }
// lldb-command:print ret
// lldb-check:[...]$8 = ((5, Struct { a: 6, b: 7.5 }), (Struct { a: 6, b: 7.5 }, 5))
// lldb-command:continue

#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

#[derive(Clone)]
struct Struct {
    a: isize,
    b: f64
}

fn dup_tup<T0: Clone, T1: Clone>(t0: &T0, t1: &T1) -> ((T0, T1), (T1, T0)) {
    let ret = ((t0.clone(), t1.clone()), (t1.clone(), t0.clone()));
    zzz(); // #break
    ret
}

fn main() {

    let _ = dup_tup(&1, &2.5f64);
    let _ = dup_tup(&3.5f64, &4_u16);
    let _ = dup_tup(&5, &Struct { a: 6, b: 7.5 });
}

fn zzz() {()}
