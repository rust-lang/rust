// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-win32
// ignore-android: FIXME(#10381)

// compile-flags:-g
// debugger:rbreak zzz
// debugger:run

// debugger:print 'lexical-scopes-in-block-expression::MUT_INT'
// check:$1 = 0

// STRUCT EXPRESSION
// debugger:finish
// debugger:print val
// check:$2 = -1
// debugger:print ten
// check:$3 = 10
// debugger:continue

// debugger:finish
// debugger:print val
// check:$4 = 11
// debugger:print 'lexical-scopes-in-block-expression::MUT_INT'
// check:$5 = 1
// debugger:print ten
// check:$6 = 10
// debugger:continue

// debugger:finish
// debugger:print val
// check:$7 = -1
// debugger:print ten
// check:$8 = 10
// debugger:continue

// FUNCTION CALL
// debugger:finish
// debugger:print val
// check:$9 = -1
// debugger:print ten
// check:$10 = 10
// debugger:continue

// debugger:finish
// debugger:print val
// check:$11 = 12
// debugger:print 'lexical-scopes-in-block-expression::MUT_INT'
// check:$12 = 2
// debugger:print ten
// check:$13 = 10
// debugger:continue

// debugger:finish
// debugger:print val
// check:$14 = -1
// debugger:print ten
// check:$15 = 10
// debugger:continue

// TUPLE EXPRESSION
// debugger:finish
// debugger:print val
// check:$16 = -1
// debugger:print ten
// check:$17 = 10
// debugger:continue

// debugger:finish
// debugger:print val
// check:$18 = 13
// debugger:print 'lexical-scopes-in-block-expression::MUT_INT'
// check:$19 = 3
// debugger:print ten
// check:$20 = 10
// debugger:continue

// debugger:finish
// debugger:print val
// check:$21 = -1
// debugger:print ten
// check:$22 = 10
// debugger:continue

// VEC EXPRESSION
// debugger:finish
// debugger:print val
// check:$23 = -1
// debugger:print ten
// check:$24 = 10
// debugger:continue

// debugger:finish
// debugger:print val
// check:$25 = 14
// debugger:print 'lexical-scopes-in-block-expression::MUT_INT'
// check:$26 = 4
// debugger:print ten
// check:$27 = 10
// debugger:continue

// debugger:finish
// debugger:print val
// check:$28 = -1
// debugger:print ten
// check:$29 = 10
// debugger:continue

// REPEAT VEC EXPRESSION
// debugger:finish
// debugger:print val
// check:$30 = -1
// debugger:print ten
// check:$31 = 10
// debugger:continue

// debugger:finish
// debugger:print val
// check:$32 = 15
// debugger:print 'lexical-scopes-in-block-expression::MUT_INT'
// check:$33 = 5
// debugger:print ten
// check:$34 = 10
// debugger:continue

// debugger:finish
// debugger:print val
// check:$35 = -1
// debugger:print ten
// check:$36 = 10
// debugger:continue

// ASSIGNMENT EXPRESSION
// debugger:finish
// debugger:print val
// check:$37 = -1
// debugger:print ten
// check:$38 = 10
// debugger:continue

// debugger:finish
// debugger:print val
// check:$39 = 16
// debugger:print 'lexical-scopes-in-block-expression::MUT_INT'
// check:$40 = 6
// debugger:print ten
// check:$41 = 10
// debugger:continue

// debugger:finish
// debugger:print val
// check:$42 = -1
// debugger:print ten
// check:$43 = 10
// debugger:continue


// ARITHMETIC EXPRESSION
// debugger:finish
// debugger:print val
// check:$44 = -1
// debugger:print ten
// check:$45 = 10
// debugger:continue

// debugger:finish
// debugger:print val
// check:$46 = 17
// debugger:print 'lexical-scopes-in-block-expression::MUT_INT'
// check:$47 = 7
// debugger:print ten
// check:$48 = 10
// debugger:continue

// debugger:finish
// debugger:print val
// check:$49 = -1
// debugger:print ten
// check:$50 = 10
// debugger:continue

// INDEX EXPRESSION
// debugger:finish
// debugger:print val
// check:$51 = -1
// debugger:print ten
// check:$52 = 10
// debugger:continue

// debugger:finish
// debugger:print val
// check:$53 = 18
// debugger:print 'lexical-scopes-in-block-expression::MUT_INT'
// check:$54 = 8
// debugger:print ten
// check:$55 = 10
// debugger:continue

// debugger:finish
// debugger:print val
// check:$56 = -1
// debugger:print ten
// check:$57 = 10
// debugger:continue

#![allow(unused_variable)]
#![allow(dead_assignment)]

static mut MUT_INT: int = 0;

struct Point {
    x: int,
    y: int
}

fn a_function(x: int) -> int {
    x + 1
}

fn main() {

    let val = -1;
    let ten = 10;

    // surrounded by struct expression
    let point = Point {
        x: {
            zzz();
            sentinel();

            let val = ten + 1;
            unsafe {MUT_INT = 1;};

            zzz();
            sentinel();

            val
        },
        y: 10
    };

    zzz();
    sentinel();

    // surrounded by function call
    let _ = a_function({
        zzz();
        sentinel();

        let val = ten + 2;
        unsafe {MUT_INT = 2;};

        zzz();
        sentinel();

        val
    });

    zzz();
    sentinel();


    // surrounded by tup
    let _ = ({
        zzz();
        sentinel();

        let val = ten + 3;
        unsafe {MUT_INT = 3;};

        zzz();
        sentinel();

        val
    }, 0);

    zzz();
    sentinel();

    // surrounded by vec
    let _ = [{
        zzz();
        sentinel();

        let val = ten + 4;
        unsafe {MUT_INT = 4;};

        zzz();
        sentinel();

        val
    }, 0, 0];

    zzz();
    sentinel();

    // surrounded by repeat vec
    let _ = [{
        zzz();
        sentinel();

        let val = ten + 5;
        unsafe {MUT_INT = 5;};

        zzz();
        sentinel();

        val
    }, ..10];

    zzz();
    sentinel();

    // assignment expression
    let mut var = 0;
    var = {
        zzz();
        sentinel();

        let val = ten + 6;
        unsafe {MUT_INT = 6;};

        zzz();
        sentinel();

        val
    };

    zzz();
    sentinel();

    // arithmetic expression
    var = 10 + -{
        zzz();
        sentinel();

        let val = ten + 7;
        unsafe {MUT_INT = 7;};

        zzz();
        sentinel();

        val
    } * 5;

    zzz();
    sentinel();

    // index expression
    let a_vector = [10, ..20];
    let _ = a_vector[{
        zzz();
        sentinel();

        let val = ten + 8;
        unsafe {MUT_INT = 8;};

        zzz();
        sentinel();

        val as uint
    }];

    zzz();
    sentinel();
}

fn zzz() {()}
fn sentinel() {()}
