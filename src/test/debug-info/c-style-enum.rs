// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:-Z extra-debug-info
// debugger:rbreak zzz
// debugger:run
// debugger:finish

// debugger:print auto_one
// check:$1 = One

// debugger:print auto_two
// check:$2 = Two

// debugger:print auto_three
// check:$3 = Three

// debugger:print manual_one_hundred
// check:$4 = OneHundred

// debugger:print manual_one_thousand
// check:$5 = OneThousand

// debugger:print manual_one_million
// check:$6 = OneMillion

// debugger:print single_variant
// check:$7 = TheOnlyVariant

#[allow(unused_variable)];

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

fn main() {

    let auto_one = One;
    let auto_two = Two;
    let auto_three = Three;

    let manual_one_hundred = OneHundred;
    let manual_one_thousand = OneThousand;
    let manual_one_million = OneMillion;

    let single_variant = TheOnlyVariant;

    zzz();
}

fn zzz() {()}
