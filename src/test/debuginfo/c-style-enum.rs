// ignore-tidy-linelength

// ignore-aarch64
// ignore-gdb // Test temporarily ignored due to debuginfo tests being disabled, see PR 47155
// min-lldb-version: 310

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdbg-command:print 'c_style_enum::SINGLE_VARIANT'
// gdbr-command:print c_style_enum::SINGLE_VARIANT
// gdbg-check:$1 = TheOnlyVariant
// gdbr-check:$1 = c_style_enum::SingleVariant::TheOnlyVariant

// gdbg-command:print 'c_style_enum::AUTO_ONE'
// gdbr-command:print c_style_enum::AUTO_ONE
// gdbg-check:$2 = One
// gdbr-check:$2 = c_style_enum::AutoDiscriminant::One

// gdbg-command:print 'c_style_enum::AUTO_TWO'
// gdbr-command:print c_style_enum::AUTO_TWO
// gdbg-check:$3 = One
// gdbr-check:$3 = c_style_enum::AutoDiscriminant::One

// gdbg-command:print 'c_style_enum::AUTO_THREE'
// gdbr-command:print c_style_enum::AUTO_THREE
// gdbg-check:$4 = One
// gdbr-check:$4 = c_style_enum::AutoDiscriminant::One

// gdbg-command:print 'c_style_enum::MANUAL_ONE'
// gdbr-command:print c_style_enum::MANUAL_ONE
// gdbg-check:$5 = OneHundred
// gdbr-check:$5 = c_style_enum::ManualDiscriminant::OneHundred

// gdbg-command:print 'c_style_enum::MANUAL_TWO'
// gdbr-command:print c_style_enum::MANUAL_TWO
// gdbg-check:$6 = OneHundred
// gdbr-check:$6 = c_style_enum::ManualDiscriminant::OneHundred

// gdbg-command:print 'c_style_enum::MANUAL_THREE'
// gdbr-command:print c_style_enum::MANUAL_THREE
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
// lldbg-check:[...]$0 = One
// lldbr-check:(c_style_enum::AutoDiscriminant) auto_one = c_style_enum::AutoDiscriminant::One

// lldb-command:print auto_two
// lldbg-check:[...]$1 = Two
// lldbr-check:(c_style_enum::AutoDiscriminant) auto_two = c_style_enum::AutoDiscriminant::Two

// lldb-command:print auto_three
// lldbg-check:[...]$2 = Three
// lldbr-check:(c_style_enum::AutoDiscriminant) auto_three = c_style_enum::AutoDiscriminant::Three

// lldb-command:print manual_one_hundred
// lldbg-check:[...]$3 = OneHundred
// lldbr-check:(c_style_enum::ManualDiscriminant) manual_one_hundred = c_style_enum::ManualDiscriminant::OneHundred

// lldb-command:print manual_one_thousand
// lldbg-check:[...]$4 = OneThousand
// lldbr-check:(c_style_enum::ManualDiscriminant) manual_one_thousand = c_style_enum::ManualDiscriminant::OneThousand

// lldb-command:print manual_one_million
// lldbg-check:[...]$5 = OneMillion
// lldbr-check:(c_style_enum::ManualDiscriminant) manual_one_million = c_style_enum::ManualDiscriminant::OneMillion

// lldb-command:print single_variant
// lldbg-check:[...]$6 = TheOnlyVariant
// lldbr-check:(c_style_enum::SingleVariant) single_variant = c_style_enum::SingleVariant::TheOnlyVariant

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
#[repr(u8)]
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
