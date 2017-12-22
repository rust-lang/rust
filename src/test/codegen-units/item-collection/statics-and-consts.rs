// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength
// compile-flags:-Zprint-trans-items=eager

#![deny(dead_code)]
#![feature(start)]

static STATIC1: i64 = {
    const STATIC1_CONST1: i64 = 2;
    1 + CONST1 as i64 + STATIC1_CONST1
};

const CONST1: i64 = {
    const CONST1_1: i64 = {
        const CONST1_1_1: i64 = 2;
        CONST1_1_1 + 1
    };
    1 + CONST1_1 as i64
};

fn foo() {
    let _ = {
        const CONST2: i64 = 0;
        static STATIC2: i64 = CONST2;

        let x = {
            const CONST2: i64 = 1;
            static STATIC2: i64 = CONST2;
            STATIC2
        };

        x + STATIC2
    };

    let _ = {
        const CONST2: i64 = 0;
        static STATIC2: i64 = CONST2;
        STATIC2
    };
}

//~ TRANS_ITEM fn statics_and_consts::start[0]
#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    foo();
    let _ = STATIC1;

    0
}

//~ TRANS_ITEM static statics_and_consts::STATIC1[0]

//~ TRANS_ITEM fn statics_and_consts::foo[0]
//~ TRANS_ITEM static statics_and_consts::foo[0]::STATIC2[0]
//~ TRANS_ITEM static statics_and_consts::foo[0]::STATIC2[1]
//~ TRANS_ITEM static statics_and_consts::foo[0]::STATIC2[2]
