// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-aarch64
// min-lldb-version: 310

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:print 'c_style_enum::SINGLE_VARIANT'
// gdbg-check:$1 = TheOnlyVariant
// gdbr-check:$1 = c_style_enum::SingleVariant::TheOnlyVariant

// gdb-command:print 'c_style_enum::AUTO_ONE'
// gdbg-check:$2 = One
// gdbr-check:$2 = c_style_enum::AutoDiscriminant::One

// gdb-command:print 'c_style_enum::AUTO_TWO'
// gdbg-check:$3 = One
// gdbr-check:$3 = c_style_enum::AutoDiscriminant::One

// gdb-command:print 'c_style_enum::AUTO_THREE'
// gdbg-check:$4 = One
// gdbr-check:$4 = c_style_enum::AutoDiscriminant::One

// gdb-command:print 'c_style_enum::MANUAL_ONE'
// gdbg-check:$5 = OneHundred
// gdbr-check:$5 = c_style_enum::ManualDiscriminant::OneHundred

// gdb-command:print 'c_style_enum::MANUAL_TWO'
// gdbg-check:$6 = OneHundred
// gdbr-check:$6 = c_style_enum::ManualDiscriminant::OneHundred

// gdb-command:print 'c_style_enum::MANUAL_THREE'
// gdbg-check:$7 = OneHundred
// gdbr-check:$7 = c_style_enum::ManualDiscriminant::OneHundred

// gdb-command:run

// gdb-command:print auto_one
// gdbg-check:$8 = One
// gdbr-check:$8 = c_style_enum::AutoDiscriminant::One

// gdb-command:print auto_two
// gdbg-check:$9 = Two
// gdbr-check:$9 = c_style_enum::AutoDiscriminant::Two

// gdb-command:print auto_three
// gdbg-check:$10 = Three
// gdbr-check:$10 = c_style_enum::AutoDiscriminant::Three

// gdb-command:print manual_one_hundred
// gdbg-check:$11 = OneHundred
// gdbr-check:$11 = c_style_enum::ManualDiscriminant::OneHundred

// gdb-command:print manual_one_thousand
// gdbg-check:$12 = OneThousand
// gdbr-check:$12 = c_style_enum::ManualDiscriminant::OneThousand

// gdb-command:print manual_one_million
// gdbg-check:$13 = OneMillion
// gdbr-check:$13 = c_style_enum::ManualDiscriminant::OneMillion

// gdb-command:print single_variant
// gdbg-check:$14 = TheOnlyVariant
// gdbr-check:$14 = c_style_enum::SingleVariant::TheOnlyVariant

// gdbg-command:print 'c_style_enum::AUTO_TWO'
// gdbr-command:print AUTO_TWO
// gdbg-check:$15 = Two
// gdbr-check:$15 = c_style_enum::AutoDiscriminant::Two

// gdbg-command:print 'c_style_enum::AUTO_THREE'
// gdbr-command:print AUTO_THREE
// gdbg-check:$16 = Three
// gdbr-check:$16 = c_style_enum::AutoDiscriminant::Three

// gdbg-command:print 'c_style_enum::MANUAL_TWO'
// gdbr-command:print MANUAL_TWO
// gdbg-check:$17 = OneThousand
// gdbr-check:$17 = c_style_enum::ManualDiscriminant::OneThousand

// gdbg-command:print 'c_style_enum::MANUAL_THREE'
// gdbr-command:print MANUAL_THREE
// gdbg-check:$18 = OneMillion
// gdbr-check:$18 = c_style_enum::ManualDiscriminant::OneMillion


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print auto_one
// lldb-check:[...]$0 = One

// lldb-command:print auto_two
// lldb-check:[...]$1 = Two

// lldb-command:print auto_three
// lldb-check:[...]$2 = Three

// lldb-command:print manual_one_hundred
// lldb-check:[...]$3 = OneHundred

// lldb-command:print manual_one_thousand
// lldb-check:[...]$4 = OneThousand

// lldb-command:print manual_one_million
// lldb-check:[...]$5 = OneMillion

// lldb-command:print single_variant
// lldb-check:[...]$6 = TheOnlyVariant

#![allow(unused_variables)]
#![allow(dead_code)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

use self::AutoDiscriminant::{One, Two, Three};
use self::ManualDiscriminant::{OneHundred, OneThousand, OneMillion};
use self::SingleVariant::TheOnlyVariant;

#[derive(Copy, Clone)]
enum AutoDiscriminant {
    One,
    Two,
    Three
}

#[derive(Copy, Clone)]
enum ManualDiscriminant {
    OneHundred = 100,
    OneThousand = 1000,
    OneMillion = 1000000
}

#[derive(Copy, Clone)]
enum SingleVariant {
    TheOnlyVariant
}

static SINGLE_VARIANT: SingleVariant = TheOnlyVariant;

static mut AUTO_ONE: AutoDiscriminant = One;
static mut AUTO_TWO: AutoDiscriminant = One;
static mut AUTO_THREE: AutoDiscriminant = One;

static mut MANUAL_ONE: ManualDiscriminant = OneHundred;
static mut MANUAL_TWO: ManualDiscriminant = OneHundred;
static mut MANUAL_THREE: ManualDiscriminant = OneHundred;

fn main() {

    let auto_one = One;
    let auto_two = Two;
    let auto_three = Three;

    let manual_one_hundred = OneHundred;
    let manual_one_thousand = OneThousand;
    let manual_one_million = OneMillion;

    let single_variant = TheOnlyVariant;

    unsafe {
        AUTO_TWO = Two;
        AUTO_THREE = Three;

        MANUAL_TWO = OneThousand;
        MANUAL_THREE = OneMillion;
    };

    zzz(); // #break

    // Borrow to avoid an eager load of the constant value in the static.
    let a = &SINGLE_VARIANT;
    let a = unsafe { AUTO_ONE };
    let a = unsafe { MANUAL_ONE };
}

fn zzz() { () }
