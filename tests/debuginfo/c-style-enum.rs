//@ ignore-aarch64

//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:print c_style_enum::SINGLE_VARIANT
// gdb-check:$1 = c_style_enum::SingleVariant::TheOnlyVariant

// gdb-command:print c_style_enum::AUTO_ONE
// gdb-check:$2 = c_style_enum::AutoDiscriminant::One

// gdb-command:print c_style_enum::AUTO_TWO
// gdb-check:$3 = c_style_enum::AutoDiscriminant::One

// gdb-command:print c_style_enum::AUTO_THREE
// gdb-check:$4 = c_style_enum::AutoDiscriminant::One

// gdb-command:print c_style_enum::MANUAL_ONE
// gdb-check:$5 = c_style_enum::ManualDiscriminant::OneHundred

// gdb-command:print c_style_enum::MANUAL_TWO
// gdb-check:$6 = c_style_enum::ManualDiscriminant::OneHundred

// gdb-command:print c_style_enum::MANUAL_THREE
// gdb-check:$7 = c_style_enum::ManualDiscriminant::OneHundred

// gdb-command:run

// gdb-command:print auto_one
// gdb-check:$8 = c_style_enum::AutoDiscriminant::One

// gdb-command:print auto_two
// gdb-check:$9 = c_style_enum::AutoDiscriminant::Two

// gdb-command:print auto_three
// gdb-check:$10 = c_style_enum::AutoDiscriminant::Three

// gdb-command:print manual_one_hundred
// gdb-check:$11 = c_style_enum::ManualDiscriminant::OneHundred

// gdb-command:print manual_one_thousand
// gdb-check:$12 = c_style_enum::ManualDiscriminant::OneThousand

// gdb-command:print manual_one_million
// gdb-check:$13 = c_style_enum::ManualDiscriminant::OneMillion

// gdb-command:print single_variant
// gdb-check:$14 = c_style_enum::SingleVariant::TheOnlyVariant

// gdb-command:print AUTO_TWO
// gdb-check:$15 = c_style_enum::AutoDiscriminant::Two

// gdb-command:print AUTO_THREE
// gdb-check:$16 = c_style_enum::AutoDiscriminant::Three

// gdb-command:print MANUAL_TWO
// gdb-check:$17 = c_style_enum::ManualDiscriminant::OneThousand

// gdb-command:print MANUAL_THREE
// gdb-check:$18 = c_style_enum::ManualDiscriminant::OneMillion


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:v auto_one
// lldb-check:[...] One

// lldb-command:v auto_two
// lldb-check:[...] Two

// lldb-command:v auto_three
// lldb-check:[...] Three

// lldb-command:v manual_one_hundred
// lldb-check:[...] OneHundred

// lldb-command:v manual_one_thousand
// lldb-check:[...] OneThousand

// lldb-command:v manual_one_million
// lldb-check:[...] OneMillion

// lldb-command:v single_variant
// lldb-check:[...] TheOnlyVariant

#![allow(unused_variables)]
#![allow(dead_code)]

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
