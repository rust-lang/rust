// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// min-lldb-version: 310

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print *stack_val_ref
// gdbg-check:$1 = {__0 = -14, __1 = -19}
// gdbr-check:$1 = (-14, -19)

// gdb-command:print *ref_to_unnamed
// gdbg-check:$2 = {__0 = -15, __1 = -20}
// gdbr-check:$2 = (-15, -20)

// gdb-command:print *unique_val_ref
// gdbg-check:$3 = {__0 = -17, __1 = -22}
// gdbr-check:$3 = (-17, -22)


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print *stack_val_ref
// lldb-check:[...]$0 = (-14, -19)

// lldb-command:print *ref_to_unnamed
// lldb-check:[...]$1 = (-15, -20)

// lldb-command:print *unique_val_ref
// lldb-check:[...]$2 = (-17, -22)


#![allow(unused_variables)]
#![feature(box_syntax)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

fn main() {
    let stack_val: (i16, f32) = (-14, -19f32);
    let stack_val_ref: &(i16, f32) = &stack_val;
    let ref_to_unnamed: &(i16, f32) = &(-15, -20f32);

    let unique_val: Box<(i16, f32)> = box (-17, -22f32);
    let unique_val_ref: &(i16, f32) = &*unique_val;

    zzz(); // #break
}

fn zzz() {()}
