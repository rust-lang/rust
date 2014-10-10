// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-android: FIXME(#10381)

// This test case checks if function arguments already have the correct value when breaking at the
// beginning of a function. Functions with the #[no_stack_check] attribute have the same prologue as
// regular C functions compiled with GCC or Clang and therefore are better handled by GDB. As a
// consequence, and as opposed to regular Rust functions, we can set the breakpoints via the
// function name (and don't have to fall back on using line numbers). For LLDB this shouldn't make
// a difference because it can handle both cases.

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:set print pretty off
// gdb-command:rbreak immediate_args
// gdb-command:rbreak binding
// gdb-command:rbreak assignment
// gdb-command:rbreak function_call
// gdb-command:rbreak identifier
// gdb-command:rbreak return_expr
// gdb-command:rbreak arithmetic_expr
// gdb-command:rbreak if_expr
// gdb-command:rbreak while_expr
// gdb-command:rbreak loop_expr
// gdb-command:run

// IMMEDIATE ARGS
// gdb-command:print a
// gdb-check:$1 = 1
// gdb-command:print b
// gdb-check:$2 = true
// gdb-command:print c
// gdb-check:$3 = 2.5
// gdb-command:continue

// NON IMMEDIATE ARGS
// gdb-command:print a
// gdb-check:$4 = {a = 3, b = 4, c = 5, d = 6, e = 7, f = 8, g = 9, h = 10}
// gdb-command:print b
// gdb-check:$5 = {a = 11, b = 12, c = 13, d = 14, e = 15, f = 16, g = 17, h = 18}
// gdb-command:continue

// BINDING
// gdb-command:print a
// gdb-check:$6 = 19
// gdb-command:print b
// gdb-check:$7 = 20
// gdb-command:print c
// gdb-check:$8 = 21.5
// gdb-command:continue

// ASSIGNMENT
// gdb-command:print a
// gdb-check:$9 = 22
// gdb-command:print b
// gdb-check:$10 = 23
// gdb-command:print c
// gdb-check:$11 = 24.5
// gdb-command:continue

// FUNCTION CALL
// gdb-command:print x
// gdb-check:$12 = 25
// gdb-command:print y
// gdb-check:$13 = 26
// gdb-command:print z
// gdb-check:$14 = 27.5
// gdb-command:continue

// EXPR
// gdb-command:print x
// gdb-check:$15 = 28
// gdb-command:print y
// gdb-check:$16 = 29
// gdb-command:print z
// gdb-check:$17 = 30.5
// gdb-command:continue

// RETURN EXPR
// gdb-command:print x
// gdb-check:$18 = 31
// gdb-command:print y
// gdb-check:$19 = 32
// gdb-command:print z
// gdb-check:$20 = 33.5
// gdb-command:continue

// ARITHMETIC EXPR
// gdb-command:print x
// gdb-check:$21 = 34
// gdb-command:print y
// gdb-check:$22 = 35
// gdb-command:print z
// gdb-check:$23 = 36.5
// gdb-command:continue

// IF EXPR
// gdb-command:print x
// gdb-check:$24 = 37
// gdb-command:print y
// gdb-check:$25 = 38
// gdb-command:print z
// gdb-check:$26 = 39.5
// gdb-command:continue

// WHILE EXPR
// gdb-command:print x
// gdb-check:$27 = 40
// gdb-command:print y
// gdb-check:$28 = 41
// gdb-command:print z
// gdb-check:$29 = 42
// gdb-command:continue

// LOOP EXPR
// gdb-command:print x
// gdb-check:$30 = 43
// gdb-command:print y
// gdb-check:$31 = 44
// gdb-command:print z
// gdb-check:$32 = 45
// gdb-command:continue


// === LLDB TESTS ==================================================================================

// lldb-command:breakpoint set --name immediate_args
// lldb-command:breakpoint set --name non_immediate_args
// lldb-command:breakpoint set --name binding
// lldb-command:breakpoint set --name assignment
// lldb-command:breakpoint set --name function_call
// lldb-command:breakpoint set --name identifier
// lldb-command:breakpoint set --name return_expr
// lldb-command:breakpoint set --name arithmetic_expr
// lldb-command:breakpoint set --name if_expr
// lldb-command:breakpoint set --name while_expr
// lldb-command:breakpoint set --name loop_expr
// lldb-command:run

// IMMEDIATE ARGS
// lldb-command:print a
// lldb-check:[...]$0 = 1
// lldb-command:print b
// lldb-check:[...]$1 = true
// lldb-command:print c
// lldb-check:[...]$2 = 2.5
// lldb-command:continue

// NON IMMEDIATE ARGS
// lldb-command:print a
// lldb-check:[...]$3 = BigStruct { a: 3, b: 4, c: 5, d: 6, e: 7, f: 8, g: 9, h: 10 }
// lldb-command:print b
// lldb-check:[...]$4 = BigStruct { a: 11, b: 12, c: 13, d: 14, e: 15, f: 16, g: 17, h: 18 }
// lldb-command:continue

// BINDING
// lldb-command:print a
// lldb-check:[...]$5 = 19
// lldb-command:print b
// lldb-check:[...]$6 = 20
// lldb-command:print c
// lldb-check:[...]$7 = 21.5
// lldb-command:continue

// ASSIGNMENT
// lldb-command:print a
// lldb-check:[...]$8 = 22
// lldb-command:print b
// lldb-check:[...]$9 = 23
// lldb-command:print c
// lldb-check:[...]$10 = 24.5
// lldb-command:continue

// FUNCTION CALL
// lldb-command:print x
// lldb-check:[...]$11 = 25
// lldb-command:print y
// lldb-check:[...]$12 = 26
// lldb-command:print z
// lldb-check:[...]$13 = 27.5
// lldb-command:continue

// EXPR
// lldb-command:print x
// lldb-check:[...]$14 = 28
// lldb-command:print y
// lldb-check:[...]$15 = 29
// lldb-command:print z
// lldb-check:[...]$16 = 30.5
// lldb-command:continue

// RETURN EXPR
// lldb-command:print x
// lldb-check:[...]$17 = 31
// lldb-command:print y
// lldb-check:[...]$18 = 32
// lldb-command:print z
// lldb-check:[...]$19 = 33.5
// lldb-command:continue

// ARITHMETIC EXPR
// lldb-command:print x
// lldb-check:[...]$20 = 34
// lldb-command:print y
// lldb-check:[...]$21 = 35
// lldb-command:print z
// lldb-check:[...]$22 = 36.5
// lldb-command:continue

// IF EXPR
// lldb-command:print x
// lldb-check:[...]$23 = 37
// lldb-command:print y
// lldb-check:[...]$24 = 38
// lldb-command:print z
// lldb-check:[...]$25 = 39.5
// lldb-command:continue

// WHILE EXPR
// lldb-command:print x
// lldb-check:[...]$26 = 40
// lldb-command:print y
// lldb-check:[...]$27 = 41
// lldb-command:print z
// lldb-check:[...]$28 = 42
// lldb-command:continue

// LOOP EXPR
// lldb-command:print x
// lldb-check:[...]$29 = 43
// lldb-command:print y
// lldb-check:[...]$30 = 44
// lldb-command:print z
// lldb-check:[...]$31 = 45
// lldb-command:continue

#![allow(unused_variable)]

#[no_stack_check]
fn immediate_args(a: int, b: bool, c: f64) {
    ()
}

struct BigStruct {
    a: u64,
    b: u64,
    c: u64,
    d: u64,
    e: u64,
    f: u64,
    g: u64,
    h: u64
}

#[no_stack_check]
fn non_immediate_args(a: BigStruct, b: BigStruct) {
    ()
}

#[no_stack_check]
fn binding(a: i64, b: u64, c: f64) {
    let x = 0i;
}

#[no_stack_check]
fn assignment(mut a: u64, b: u64, c: f64) {
    a = b;
}

#[no_stack_check]
fn function_call(x: u64, y: u64, z: f64) {
    std::io::stdio::print("Hi!")
}

#[no_stack_check]
fn identifier(x: u64, y: u64, z: f64) -> u64 {
    x
}

#[no_stack_check]
fn return_expr(x: u64, y: u64, z: f64) -> u64 {
    return x;
}

#[no_stack_check]
fn arithmetic_expr(x: u64, y: u64, z: f64) -> u64 {
    x + y
}

#[no_stack_check]
fn if_expr(x: u64, y: u64, z: f64) -> u64 {
    if x + y < 1000 {
        x
    } else {
        y
    }
}

#[no_stack_check]
fn while_expr(mut x: u64, y: u64, z: u64) -> u64 {
    while x + y < 1000 {
        x += z
    }
    return x;
}

#[no_stack_check]
fn loop_expr(mut x: u64, y: u64, z: u64) -> u64 {
    loop {
        x += z;

        if x + y > 1000 {
            return x;
        }
    }
}

fn main() {
    immediate_args(1, true, 2.5);

    non_immediate_args(
        BigStruct {
            a: 3,
            b: 4,
            c: 5,
            d: 6,
            e: 7,
            f: 8,
            g: 9,
            h: 10
        },
        BigStruct {
            a: 11,
            b: 12,
            c: 13,
            d: 14,
            e: 15,
            f: 16,
            g: 17,
            h: 18
        }
    );

    binding(19, 20, 21.5);
    assignment(22, 23, 24.5);
    function_call(25, 26, 27.5);
    identifier(28, 29, 30.5);
    return_expr(31, 32, 33.5);
    arithmetic_expr(34, 35, 36.5);
    if_expr(37, 38, 39.5);
    while_expr(40, 41, 42);
    loop_expr(43, 44, 45);
}
