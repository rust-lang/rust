// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-windows: FIXME #13256
// ignore-android: FIXME(#10381)

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:rbreak zzz
// gdb-command:run

// gdb-command:print 'lexical-scopes-in-block-expression::MUT_INT'
// gdb-check:$1 = 0

// STRUCT EXPRESSION
// gdb-command:finish
// gdb-command:print val
// gdb-check:$2 = -1
// gdb-command:print ten
// gdb-check:$3 = 10
// gdb-command:continue

// gdb-command:finish
// gdb-command:print val
// gdb-check:$4 = 11
// gdb-command:print 'lexical-scopes-in-block-expression::MUT_INT'
// gdb-check:$5 = 1
// gdb-command:print ten
// gdb-check:$6 = 10
// gdb-command:continue

// gdb-command:finish
// gdb-command:print val
// gdb-check:$7 = -1
// gdb-command:print ten
// gdb-check:$8 = 10
// gdb-command:continue

// FUNCTION CALL
// gdb-command:finish
// gdb-command:print val
// gdb-check:$9 = -1
// gdb-command:print ten
// gdb-check:$10 = 10
// gdb-command:continue

// gdb-command:finish
// gdb-command:print val
// gdb-check:$11 = 12
// gdb-command:print 'lexical-scopes-in-block-expression::MUT_INT'
// gdb-check:$12 = 2
// gdb-command:print ten
// gdb-check:$13 = 10
// gdb-command:continue

// gdb-command:finish
// gdb-command:print val
// gdb-check:$14 = -1
// gdb-command:print ten
// gdb-check:$15 = 10
// gdb-command:continue

// TUPLE EXPRESSION
// gdb-command:finish
// gdb-command:print val
// gdb-check:$16 = -1
// gdb-command:print ten
// gdb-check:$17 = 10
// gdb-command:continue

// gdb-command:finish
// gdb-command:print val
// gdb-check:$18 = 13
// gdb-command:print 'lexical-scopes-in-block-expression::MUT_INT'
// gdb-check:$19 = 3
// gdb-command:print ten
// gdb-check:$20 = 10
// gdb-command:continue

// gdb-command:finish
// gdb-command:print val
// gdb-check:$21 = -1
// gdb-command:print ten
// gdb-check:$22 = 10
// gdb-command:continue

// VEC EXPRESSION
// gdb-command:finish
// gdb-command:print val
// gdb-check:$23 = -1
// gdb-command:print ten
// gdb-check:$24 = 10
// gdb-command:continue

// gdb-command:finish
// gdb-command:print val
// gdb-check:$25 = 14
// gdb-command:print 'lexical-scopes-in-block-expression::MUT_INT'
// gdb-check:$26 = 4
// gdb-command:print ten
// gdb-check:$27 = 10
// gdb-command:continue

// gdb-command:finish
// gdb-command:print val
// gdb-check:$28 = -1
// gdb-command:print ten
// gdb-check:$29 = 10
// gdb-command:continue

// REPEAT VEC EXPRESSION
// gdb-command:finish
// gdb-command:print val
// gdb-check:$30 = -1
// gdb-command:print ten
// gdb-check:$31 = 10
// gdb-command:continue

// gdb-command:finish
// gdb-command:print val
// gdb-check:$32 = 15
// gdb-command:print 'lexical-scopes-in-block-expression::MUT_INT'
// gdb-check:$33 = 5
// gdb-command:print ten
// gdb-check:$34 = 10
// gdb-command:continue

// gdb-command:finish
// gdb-command:print val
// gdb-check:$35 = -1
// gdb-command:print ten
// gdb-check:$36 = 10
// gdb-command:continue

// ASSIGNMENT EXPRESSION
// gdb-command:finish
// gdb-command:print val
// gdb-check:$37 = -1
// gdb-command:print ten
// gdb-check:$38 = 10
// gdb-command:continue

// gdb-command:finish
// gdb-command:print val
// gdb-check:$39 = 16
// gdb-command:print 'lexical-scopes-in-block-expression::MUT_INT'
// gdb-check:$40 = 6
// gdb-command:print ten
// gdb-check:$41 = 10
// gdb-command:continue

// gdb-command:finish
// gdb-command:print val
// gdb-check:$42 = -1
// gdb-command:print ten
// gdb-check:$43 = 10
// gdb-command:continue


// ARITHMETIC EXPRESSION
// gdb-command:finish
// gdb-command:print val
// gdb-check:$44 = -1
// gdb-command:print ten
// gdb-check:$45 = 10
// gdb-command:continue

// gdb-command:finish
// gdb-command:print val
// gdb-check:$46 = 17
// gdb-command:print 'lexical-scopes-in-block-expression::MUT_INT'
// gdb-check:$47 = 7
// gdb-command:print ten
// gdb-check:$48 = 10
// gdb-command:continue

// gdb-command:finish
// gdb-command:print val
// gdb-check:$49 = -1
// gdb-command:print ten
// gdb-check:$50 = 10
// gdb-command:continue

// INDEX EXPRESSION
// gdb-command:finish
// gdb-command:print val
// gdb-check:$51 = -1
// gdb-command:print ten
// gdb-check:$52 = 10
// gdb-command:continue

// gdb-command:finish
// gdb-command:print val
// gdb-check:$53 = 18
// gdb-command:print 'lexical-scopes-in-block-expression::MUT_INT'
// gdb-check:$54 = 8
// gdb-command:print ten
// gdb-check:$55 = 10
// gdb-command:continue

// gdb-command:finish
// gdb-command:print val
// gdb-check:$56 = -1
// gdb-command:print ten
// gdb-check:$57 = 10
// gdb-command:continue


// === LLDB TESTS ==================================================================================

// lldb-command:run

// STRUCT EXPRESSION
// lldb-command:print val
// lldb-check:[...]$0 = -1
// lldb-command:print ten
// lldb-check:[...]$1 = 10
// lldb-command:continue

// lldb-command:print val
// lldb-check:[...]$2 = 11
// lldb-command:print ten
// lldb-check:[...]$3 = 10
// lldb-command:continue

// lldb-command:print val
// lldb-check:[...]$4 = -1
// lldb-command:print ten
// lldb-check:[...]$5 = 10
// lldb-command:continue

// FUNCTION CALL
// lldb-command:print val
// lldb-check:[...]$6 = -1
// lldb-command:print ten
// lldb-check:[...]$7 = 10
// lldb-command:continue

// lldb-command:print val
// lldb-check:[...]$8 = 12
// lldb-command:print ten
// lldb-check:[...]$9 = 10
// lldb-command:continue

// lldb-command:print val
// lldb-check:[...]$10 = -1
// lldb-command:print ten
// lldb-check:[...]$11 = 10
// lldb-command:continue

// TUPLE EXPRESSION
// lldb-command:print val
// lldb-check:[...]$12 = -1
// lldb-command:print ten
// lldb-check:[...]$13 = 10
// lldb-command:continue

// lldb-command:print val
// lldb-check:[...]$14 = 13
// lldb-command:print ten
// lldb-check:[...]$15 = 10
// lldb-command:continue

// lldb-command:print val
// lldb-check:[...]$16 = -1
// lldb-command:print ten
// lldb-check:[...]$17 = 10
// lldb-command:continue

// VEC EXPRESSION
// lldb-command:print val
// lldb-check:[...]$18 = -1
// lldb-command:print ten
// lldb-check:[...]$19 = 10
// lldb-command:continue

// lldb-command:print val
// lldb-check:[...]$20 = 14
// lldb-command:print ten
// lldb-check:[...]$21 = 10
// lldb-command:continue

// lldb-command:print val
// lldb-check:[...]$22 = -1
// lldb-command:print ten
// lldb-check:[...]$23 = 10
// lldb-command:continue

// REPEAT VEC EXPRESSION
// lldb-command:print val
// lldb-check:[...]$24 = -1
// lldb-command:print ten
// lldb-check:[...]$25 = 10
// lldb-command:continue

// lldb-command:print val
// lldb-check:[...]$26 = 15
// lldb-command:print ten
// lldb-check:[...]$27 = 10
// lldb-command:continue

// lldb-command:print val
// lldb-check:[...]$28 = -1
// lldb-command:print ten
// lldb-check:[...]$29 = 10
// lldb-command:continue

// ASSIGNMENT EXPRESSION
// lldb-command:print val
// lldb-check:[...]$30 = -1
// lldb-command:print ten
// lldb-check:[...]$31 = 10
// lldb-command:continue

// lldb-command:print val
// lldb-check:[...]$32 = 16
// lldb-command:print ten
// lldb-check:[...]$33 = 10
// lldb-command:continue

// lldb-command:print val
// lldb-check:[...]$34 = -1
// lldb-command:print ten
// lldb-check:[...]$35 = 10
// lldb-command:continue


// ARITHMETIC EXPRESSION
// lldb-command:print val
// lldb-check:[...]$36 = -1
// lldb-command:print ten
// lldb-check:[...]$37 = 10
// lldb-command:continue

// lldb-command:print val
// lldb-check:[...]$38 = 17
// lldb-command:print ten
// lldb-check:[...]$39 = 10
// lldb-command:continue

// lldb-command:print val
// lldb-check:[...]$40 = -1
// lldb-command:print ten
// lldb-check:[...]$41 = 10
// lldb-command:continue

// INDEX EXPRESSION
// lldb-command:print val
// lldb-check:[...]$42 = -1
// lldb-command:print ten
// lldb-check:[...]$43 = 10
// lldb-command:continue

// lldb-command:print val
// lldb-check:[...]$44 = 18
// lldb-command:print ten
// lldb-check:[...]$45 = 10
// lldb-command:continue

// lldb-command:print val
// lldb-check:[...]$46 = -1
// lldb-command:print ten
// lldb-check:[...]$47 = 10
// lldb-command:continue

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

    let val = -1i;
    let ten = 10i;

    // surrounded by struct expression
    let point = Point {
        x: {
            zzz(); // #break
            sentinel();

            let val = ten + 1;
            unsafe {MUT_INT = 1;};

            zzz(); // #break
            sentinel();

            val
        },
        y: 10
    };

    zzz(); // #break
    sentinel();

    // surrounded by function call
    let _ = a_function({
        zzz(); // #break
        sentinel();

        let val = ten + 2;
        unsafe {MUT_INT = 2;};

        zzz(); // #break
        sentinel();

        val
    });

    zzz(); // #break
    sentinel();


    // surrounded by tup
    let _ = ({
        zzz(); // #break
        sentinel();

        let val = ten + 3;
        unsafe {MUT_INT = 3;};

        zzz(); // #break
        sentinel();

        val
    }, 0i);

    zzz(); // #break
    sentinel();

    // surrounded by vec
    let _ = [{
        zzz(); // #break
        sentinel();

        let val = ten + 4;
        unsafe {MUT_INT = 4;};

        zzz(); // #break
        sentinel();

        val
    }, 0, 0];

    zzz(); // #break
    sentinel();

    // surrounded by repeat vec
    let _ = [{
        zzz(); // #break
        sentinel();

        let val = ten + 5;
        unsafe {MUT_INT = 5;};

        zzz(); // #break
        sentinel();

        val
    }, ..10];

    zzz(); // #break
    sentinel();

    // assignment expression
    let mut var = 0;
    var = {
        zzz(); // #break
        sentinel();

        let val = ten + 6;
        unsafe {MUT_INT = 6;};

        zzz(); // #break
        sentinel();

        val
    };

    zzz(); // #break
    sentinel();

    // arithmetic expression
    var = 10 + -{
        zzz(); // #break
        sentinel();

        let val = ten + 7;
        unsafe {MUT_INT = 7;};

        zzz(); // #break
        sentinel();

        val
    } * 5;

    zzz(); // #break
    sentinel();

    // index expression
    let a_vector = [10i, ..20];
    let _ = a_vector[{
        zzz(); // #break
        sentinel();

        let val = ten + 8;
        unsafe {MUT_INT = 8;};

        zzz(); // #break
        sentinel();

        val as uint
    }];

    zzz(); // #break
    sentinel();
}

fn zzz() {()}
fn sentinel() {()}
