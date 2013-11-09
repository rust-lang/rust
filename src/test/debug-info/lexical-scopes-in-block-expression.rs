// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-win32
// xfail-android: FIXME(#10381)

// compile-flags:-Z extra-debug-info
// debugger:rbreak zzz
// debugger:run

// STRUCT EXPRESSION
// debugger:finish
// debugger:print val
// check:$1 = -1
// debugger:print ten
// check:$2 = 10
// debugger:continue

// debugger:finish
// debugger:print val
// check:$3 = 11
// debugger:print ten
// check:$4 = 10
// debugger:continue

// debugger:finish
// debugger:print val
// check:$5 = -1
// debugger:print ten
// check:$6 = 10
// debugger:continue

// FUNCTION CALL
// debugger:finish
// debugger:print val
// check:$7 = -1
// debugger:print ten
// check:$8 = 10
// debugger:continue

// debugger:finish
// debugger:print val
// check:$9 = 12
// debugger:print ten
// check:$10 = 10
// debugger:continue

// debugger:finish
// debugger:print val
// check:$11 = -1
// debugger:print ten
// check:$12 = 10
// debugger:continue

// TUPLE EXPRESSION
// debugger:finish
// debugger:print val
// check:$13 = -1
// debugger:print ten
// check:$14 = 10
// debugger:continue

// debugger:finish
// debugger:print val
// check:$15 = 13
// debugger:print ten
// check:$16 = 10
// debugger:continue

// debugger:finish
// debugger:print val
// check:$17 = -1
// debugger:print ten
// check:$18 = 10
// debugger:continue

// VEC EXPRESSION
// debugger:finish
// debugger:print val
// check:$19 = -1
// debugger:print ten
// check:$20 = 10
// debugger:continue

// debugger:finish
// debugger:print val
// check:$21 = 14
// debugger:print ten
// check:$22 = 10
// debugger:continue

// debugger:finish
// debugger:print val
// check:$23 = -1
// debugger:print ten
// check:$24 = 10
// debugger:continue

// REPEAT VEC EXPRESSION
// debugger:finish
// debugger:print val
// check:$25 = -1
// debugger:print ten
// check:$26 = 10
// debugger:continue

// debugger:finish
// debugger:print val
// check:$27 = 15
// debugger:print ten
// check:$28 = 10
// debugger:continue

// debugger:finish
// debugger:print val
// check:$29 = -1
// debugger:print ten
// check:$30 = 10
// debugger:continue

// ASSIGNMENT EXPRESSION
// debugger:finish
// debugger:print val
// check:$31 = -1
// debugger:print ten
// check:$32 = 10
// debugger:continue

// debugger:finish
// debugger:print val
// check:$33 = 16
// debugger:print ten
// check:$34 = 10
// debugger:continue

// debugger:finish
// debugger:print val
// check:$35 = -1
// debugger:print ten
// check:$36 = 10
// debugger:continue


// ARITHMETIC EXPRESSION
// debugger:finish
// debugger:print val
// check:$37 = -1
// debugger:print ten
// check:$38 = 10
// debugger:continue

// debugger:finish
// debugger:print val
// check:$39 = 17
// debugger:print ten
// check:$40 = 10
// debugger:continue

// debugger:finish
// debugger:print val
// check:$41 = -1
// debugger:print ten
// check:$42 = 10
// debugger:continue

// INDEX EXPRESSION
// debugger:finish
// debugger:print val
// check:$43 = -1
// debugger:print ten
// check:$44 = 10
// debugger:continue

// debugger:finish
// debugger:print val
// check:$45 = 18
// debugger:print ten
// check:$46 = 10
// debugger:continue

// debugger:finish
// debugger:print val
// check:$47 = -1
// debugger:print ten
// check:$48 = 10
// debugger:continue

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

        zzz();
        sentinel();

        val
    }];

    zzz();
    sentinel();
}

fn zzz() {()}
fn sentinel() {()}
