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
// debugger:rbreak zzz

// debugger:print 'c-style-enum::SINGLE_VARIANT'
// check:$1 = TheOnlyVariant

// debugger:print 'c-style-enum::AUTO_ONE'
// check:$2 = One

// debugger:print 'c-style-enum::AUTO_TWO'
// check:$3 = One

// debugger:print 'c-style-enum::AUTO_THREE'
// check:$4 = One

// debugger:print 'c-style-enum::MANUAL_ONE'
// check:$5 = OneHundred

// debugger:print 'c-style-enum::MANUAL_TWO'
// check:$6 = OneHundred

// debugger:print 'c-style-enum::MANUAL_THREE'
// check:$7 = OneHundred

// debugger:run
// debugger:finish

// debugger:print auto_one
// check:$8 = One

// debugger:print auto_two
// check:$9 = Two

// debugger:print auto_three
// check:$10 = Three

// debugger:print manual_one_hundred
// check:$11 = OneHundred

// debugger:print manual_one_thousand
// check:$12 = OneThousand

// debugger:print manual_one_million
// check:$13 = OneMillion

// debugger:print single_variant
// check:$14 = TheOnlyVariant

// debugger:print 'c-style-enum::AUTO_TWO'
// check:$15 = Two

// debugger:print 'c-style-enum::AUTO_THREE'
// check:$16 = Three

// debugger:print 'c-style-enum::MANUAL_TWO'
// check:$17 = OneThousand

// debugger:print 'c-style-enum::MANUAL_THREE'
// check:$18 = OneMillion

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
}

fn zzz() {()}
