// This test case checks if function arguments already have the correct value
// when breaking at the first line of the function, that is if the function
// prologue has already been executed at the first line. Note that because of
// the __morestack part of the prologue GDB incorrectly breaks at before the
// arguments have been properly loaded when setting the breakpoint via the
// function name.

//@ min-lldb-version: 1800
//@ compile-flags:-g -Zmir-enable-passes=-SingleUseConsts
// SingleUseConsts shouldn't need to be disabled, see #128945
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

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
// gdbt-check:$4 = function_arg_initialization::BigStruct {a: 3, b: 4, c: 5, d: 6, e: 7, f: 8, g: 9, h: 10}
// gdb-command:print b
// gdbt-check:$5 = function_arg_initialization::BigStruct {a: 11, b: 12, c: 13, d: 14, e: 15, f: 16, g: 17, h: 18}
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

// lldb-command:run

// IMMEDIATE ARGS
// lldb-command:v a
// lldb-check:[...] 1
// lldb-command:v b
// lldb-check:[...] true
// lldb-command:v c
// lldb-check:[...] 2.5
// lldb-command:continue

// NON IMMEDIATE ARGS
// lldb-command:v a
// lldb-check:[...] BigStruct { a: 3, b: 4, c: 5, d: 6, e: 7, f: 8, g: 9, h: 10 }
// lldb-command:v b
// lldb-check:[...] BigStruct { a: 11, b: 12, c: 13, d: 14, e: 15, f: 16, g: 17, h: 18 }
// lldb-command:continue

// BINDING
// lldb-command:v a
// lldb-check:[...] 19
// lldb-command:v b
// lldb-check:[...] 20
// lldb-command:v c
// lldb-check:[...] 21.5
// lldb-command:continue

// ASSIGNMENT
// lldb-command:v a
// lldb-check:[...] 22
// lldb-command:v b
// lldb-check:[...] 23
// lldb-command:v c
// lldb-check:[...] 24.5
// lldb-command:continue

// FUNCTION CALL
// lldb-command:v x
// lldb-check:[...] 25
// lldb-command:v y
// lldb-check:[...] 26
// lldb-command:v z
// lldb-check:[...] 27.5
// lldb-command:continue

// EXPR
// lldb-command:v x
// lldb-check:[...] 28
// lldb-command:v y
// lldb-check:[...] 29
// lldb-command:v z
// lldb-check:[...] 30.5
// lldb-command:continue

// RETURN EXPR
// lldb-command:v x
// lldb-check:[...] 31
// lldb-command:v y
// lldb-check:[...] 32
// lldb-command:v z
// lldb-check:[...] 33.5
// lldb-command:continue

// ARITHMETIC EXPR
// lldb-command:v x
// lldb-check:[...] 34
// lldb-command:v y
// lldb-check:[...] 35
// lldb-command:v z
// lldb-check:[...] 36.5
// lldb-command:continue

// IF EXPR
// lldb-command:v x
// lldb-check:[...] 37
// lldb-command:v y
// lldb-check:[...] 38
// lldb-command:v z
// lldb-check:[...] 39.5
// lldb-command:continue

// WHILE EXPR
// lldb-command:v x
// lldb-check:[...] 40
// lldb-command:v y
// lldb-check:[...] 41
// lldb-command:v z
// lldb-check:[...] 42
// lldb-command:continue

// LOOP EXPR
// lldb-command:v x
// lldb-check:[...] 43
// lldb-command:v y
// lldb-check:[...] 44
// lldb-command:v z
// lldb-check:[...] 45
// lldb-command:continue


#![allow(unused_variables)]

fn immediate_args(a: isize, b: bool, c: f64) {
    zzz(); // #break
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
    zzz(); // #break
}

fn binding(a: i64, b: u64, c: f64) {
    let x = 0; // #break
    println!()
}

fn assignment(mut a: u64, b: u64, c: f64) {
    a = b; // #break
    println!()
}

fn function_call(x: u64, y: u64, z: f64) {
    zzz(); // #break
}

fn identifier(x: u64, y: u64, z: f64) -> u64 {
    x // #break
}

fn return_expr(x: u64, y: u64, z: f64) -> u64 {
    return x; // #break
}

fn arithmetic_expr(x: u64, y: u64, z: f64) -> u64 {
    x + y // #break
}

fn if_expr(x: u64, y: u64, z: f64) -> u64 {
    if x + y < 1000 { // #break
        x
    } else {
        y
    }
}

fn while_expr(mut x: u64, y: u64, z: u64) -> u64 {
    while x + y > 1000 { // #break
        x += z
    }
    return x;
}

fn loop_expr(mut x: u64, y: u64, z: u64) -> u64 {
    loop {
        x += z; // #break

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

fn zzz() {()}
