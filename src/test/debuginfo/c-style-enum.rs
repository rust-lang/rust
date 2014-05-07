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
// gdb-command:rbreak zzz

// gdb-command:print 'c-style-enum::SINGLE_VARIANT'
// gdb-check:$1 = TheOnlyVariant

// gdb-command:print 'c-style-enum::AUTO_ONE'
// gdb-check:$2 = One

// gdb-command:print 'c-style-enum::AUTO_TWO'
// gdb-check:$3 = One

// gdb-command:print 'c-style-enum::AUTO_THREE'
// gdb-check:$4 = One

// gdb-command:print 'c-style-enum::MANUAL_ONE'
// gdb-check:$5 = OneHundred

// gdb-command:print 'c-style-enum::MANUAL_TWO'
// gdb-check:$6 = OneHundred

// gdb-command:print 'c-style-enum::MANUAL_THREE'
// gdb-check:$7 = OneHundred

// gdb-command:run
// gdb-command:finish

// gdb-command:print auto_one
// gdb-check:$8 = One

// gdb-command:print auto_two
// gdb-check:$9 = Two

// gdb-command:print auto_three
// gdb-check:$10 = Three

// gdb-command:print manual_one_hundred
// gdb-check:$11 = OneHundred

// gdb-command:print manual_one_thousand
// gdb-check:$12 = OneThousand

// gdb-command:print manual_one_million
// gdb-check:$13 = OneMillion

// gdb-command:print single_variant
// gdb-check:$14 = TheOnlyVariant

// gdb-command:print 'c-style-enum::AUTO_TWO'
// gdb-check:$15 = Two

// gdb-command:print 'c-style-enum::AUTO_THREE'
// gdb-check:$16 = Three

// gdb-command:print 'c-style-enum::MANUAL_TWO'
// gdb-check:$17 = OneThousand

// gdb-command:print 'c-style-enum::MANUAL_THREE'
// gdb-check:$18 = OneMillion

#![allow(unused_variable)]
#![allow(dead_code)]

enum AutoDiscriminant {
    One,
    Two,
    Three
}

enum ManualDiscriminant {
    OneHundred = 100,
    OneThousand = 1000,
    OneMillion = 1000000
}

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

    zzz();

    let a = SINGLE_VARIANT;
    let a = unsafe { AUTO_ONE };
    let a = unsafe { MANUAL_ONE };
}

fn zzz() {()}
