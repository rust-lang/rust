// This test case checks if function arguments already have the correct value when breaking at the
// beginning of a function.

// min-lldb-version: 310
// ignore-gdb
// ignore-test // Test temporarily ignored due to debuginfo tests being disabled, see PR 47155
// compile-flags:-g

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

#![allow(unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

fn immediate_args(a: isize, b: bool, c: f64) {
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

fn non_immediate_args(a: BigStruct, b: BigStruct) {
    ()
}

fn binding(a: i64, b: u64, c: f64) {
    let x = 0;
}

fn assignment(mut a: u64, b: u64, c: f64) {
    a = b;
}

fn function_call(x: u64, y: u64, z: f64) {
    println!("Hi!")
}

fn identifier(x: u64, y: u64, z: f64) -> u64 {
    x
}

fn return_expr(x: u64, y: u64, z: f64) -> u64 {
    return x;
}

fn arithmetic_expr(x: u64, y: u64, z: f64) -> u64 {
    x + y
}

fn if_expr(x: u64, y: u64, z: f64) -> u64 {
    if x + y < 1000 {
        x
    } else {
        y
    }
}

fn while_expr(mut x: u64, y: u64, z: u64) -> u64 {
    while x + y < 1000 {
        x += z
    }
    return x;
}

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
