//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print lexical_scopes_in_block_expression::MUT_INT
// gdb-check:$1 = 0

// STRUCT EXPRESSION
// gdb-command:print val
// gdb-check:$2 = -1
// gdb-command:print ten
// gdb-check:$3 = 10
// gdb-command:continue

// gdb-command:print val
// gdb-check:$4 = 11
// gdb-command:print lexical_scopes_in_block_expression::MUT_INT
// gdb-check:$5 = 1
// gdb-command:print ten
// gdb-check:$6 = 10
// gdb-command:continue

// gdb-command:print val
// gdb-check:$7 = -1
// gdb-command:print ten
// gdb-check:$8 = 10
// gdb-command:continue

// FUNCTION CALL
// gdb-command:print val
// gdb-check:$9 = -1
// gdb-command:print ten
// gdb-check:$10 = 10
// gdb-command:continue

// gdb-command:print val
// gdb-check:$11 = 12
// gdb-command:print lexical_scopes_in_block_expression::MUT_INT
// gdb-check:$12 = 2
// gdb-command:print ten
// gdb-check:$13 = 10
// gdb-command:continue

// gdb-command:print val
// gdb-check:$14 = -1
// gdb-command:print ten
// gdb-check:$15 = 10
// gdb-command:continue

// TUPLE EXPRESSION
// gdb-command:print val
// gdb-check:$16 = -1
// gdb-command:print ten
// gdb-check:$17 = 10
// gdb-command:continue

// gdb-command:print val
// gdb-check:$18 = 13
// gdb-command:print lexical_scopes_in_block_expression::MUT_INT
// gdb-check:$19 = 3
// gdb-command:print ten
// gdb-check:$20 = 10
// gdb-command:continue

// gdb-command:print val
// gdb-check:$21 = -1
// gdb-command:print ten
// gdb-check:$22 = 10
// gdb-command:continue

// VEC EXPRESSION
// gdb-command:print val
// gdb-check:$23 = -1
// gdb-command:print ten
// gdb-check:$24 = 10
// gdb-command:continue

// gdb-command:print val
// gdb-check:$25 = 14
// gdb-command:print lexical_scopes_in_block_expression::MUT_INT
// gdb-check:$26 = 4
// gdb-command:print ten
// gdb-check:$27 = 10
// gdb-command:continue

// gdb-command:print val
// gdb-check:$28 = -1
// gdb-command:print ten
// gdb-check:$29 = 10
// gdb-command:continue

// REPEAT VEC EXPRESSION
// gdb-command:print val
// gdb-check:$30 = -1
// gdb-command:print ten
// gdb-check:$31 = 10
// gdb-command:continue

// gdb-command:print val
// gdb-check:$32 = 15
// gdb-command:print lexical_scopes_in_block_expression::MUT_INT
// gdb-check:$33 = 5
// gdb-command:print ten
// gdb-check:$34 = 10
// gdb-command:continue

// gdb-command:print val
// gdb-check:$35 = -1
// gdb-command:print ten
// gdb-check:$36 = 10
// gdb-command:continue

// ASSIGNMENT EXPRESSION
// gdb-command:print val
// gdb-check:$37 = -1
// gdb-command:print ten
// gdb-check:$38 = 10
// gdb-command:continue

// gdb-command:print val
// gdb-check:$39 = 16
// gdb-command:print lexical_scopes_in_block_expression::MUT_INT
// gdb-check:$40 = 6
// gdb-command:print ten
// gdb-check:$41 = 10
// gdb-command:continue

// gdb-command:print val
// gdb-check:$42 = -1
// gdb-command:print ten
// gdb-check:$43 = 10
// gdb-command:continue


// ARITHMETIC EXPRESSION
// gdb-command:print val
// gdb-check:$44 = -1
// gdb-command:print ten
// gdb-check:$45 = 10
// gdb-command:continue

// gdb-command:print val
// gdb-check:$46 = 17
// gdb-command:print lexical_scopes_in_block_expression::MUT_INT
// gdb-check:$47 = 7
// gdb-command:print ten
// gdb-check:$48 = 10
// gdb-command:continue

// gdb-command:print val
// gdb-check:$49 = -1
// gdb-command:print ten
// gdb-check:$50 = 10
// gdb-command:continue

// INDEX EXPRESSION
// gdb-command:print val
// gdb-check:$51 = -1
// gdb-command:print ten
// gdb-check:$52 = 10
// gdb-command:continue

// gdb-command:print val
// gdb-check:$53 = 18
// gdb-command:print lexical_scopes_in_block_expression::MUT_INT
// gdb-check:$54 = 8
// gdb-command:print ten
// gdb-check:$55 = 10
// gdb-command:continue

// gdb-command:print val
// gdb-check:$56 = -1
// gdb-command:print ten
// gdb-check:$57 = 10
// gdb-command:continue


// === LLDB TESTS ==================================================================================

// lldb-command:run

// STRUCT EXPRESSION
// lldb-command:v val
// lldb-check:[...] -1
// lldb-command:v ten
// lldb-check:[...] 10
// lldb-command:continue

// lldb-command:v val
// lldb-check:[...] 11
// lldb-command:v ten
// lldb-check:[...] 10
// lldb-command:continue

// lldb-command:v val
// lldb-check:[...] -1
// lldb-command:v ten
// lldb-check:[...] 10
// lldb-command:continue

// FUNCTION CALL
// lldb-command:v val
// lldb-check:[...] -1
// lldb-command:v ten
// lldb-check:[...] 10
// lldb-command:continue

// lldb-command:v val
// lldb-check:[...] 12
// lldb-command:v ten
// lldb-check:[...] 10
// lldb-command:continue

// lldb-command:v val
// lldb-check:[...] -1
// lldb-command:v ten
// lldb-check:[...] 10
// lldb-command:continue

// TUPLE EXPRESSION
// lldb-command:v val
// lldb-check:[...] -1
// lldb-command:v ten
// lldb-check:[...] 10
// lldb-command:continue

// lldb-command:v val
// lldb-check:[...] 13
// lldb-command:v ten
// lldb-check:[...] 10
// lldb-command:continue

// lldb-command:v val
// lldb-check:[...] -1
// lldb-command:v ten
// lldb-check:[...] 10
// lldb-command:continue

// VEC EXPRESSION
// lldb-command:v val
// lldb-check:[...] -1
// lldb-command:v ten
// lldb-check:[...] 10
// lldb-command:continue

// lldb-command:v val
// lldb-check:[...] 14
// lldb-command:v ten
// lldb-check:[...] 10
// lldb-command:continue

// lldb-command:v val
// lldb-check:[...] -1
// lldb-command:v ten
// lldb-check:[...] 10
// lldb-command:continue

// REPEAT VEC EXPRESSION
// lldb-command:v val
// lldb-check:[...] -1
// lldb-command:v ten
// lldb-check:[...] 10
// lldb-command:continue

// lldb-command:v val
// lldb-check:[...] 15
// lldb-command:v ten
// lldb-check:[...] 10
// lldb-command:continue

// lldb-command:v val
// lldb-check:[...] -1
// lldb-command:v ten
// lldb-check:[...] 10
// lldb-command:continue

// ASSIGNMENT EXPRESSION
// lldb-command:v val
// lldb-check:[...] -1
// lldb-command:v ten
// lldb-check:[...] 10
// lldb-command:continue

// lldb-command:v val
// lldb-check:[...] 16
// lldb-command:v ten
// lldb-check:[...] 10
// lldb-command:continue

// lldb-command:v val
// lldb-check:[...] -1
// lldb-command:v ten
// lldb-check:[...] 10
// lldb-command:continue


// ARITHMETIC EXPRESSION
// lldb-command:v val
// lldb-check:[...] -1
// lldb-command:v ten
// lldb-check:[...] 10
// lldb-command:continue

// lldb-command:v val
// lldb-check:[...] 17
// lldb-command:v ten
// lldb-check:[...] 10
// lldb-command:continue

// lldb-command:v val
// lldb-check:[...] -1
// lldb-command:v ten
// lldb-check:[...] 10
// lldb-command:continue

// INDEX EXPRESSION
// lldb-command:v val
// lldb-check:[...] -1
// lldb-command:v ten
// lldb-check:[...] 10
// lldb-command:continue

// lldb-command:v val
// lldb-check:[...] 18
// lldb-command:v ten
// lldb-check:[...] 10
// lldb-command:continue

// lldb-command:v val
// lldb-check:[...] -1
// lldb-command:v ten
// lldb-check:[...] 10
// lldb-command:continue

#![allow(unused_variables)]
#![allow(unused_assignments)]

static mut MUT_INT: isize = 0;

struct Point {
    x: isize,
    y: isize
}

fn a_function(x: isize) -> isize {
    x + 1
}

fn main() {

    let val = -1;
    let ten = 10;

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
    }, 0);

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
    }; 10];

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
    let a_vector = [10; 20];
    let _ = a_vector[{
        zzz(); // #break
        sentinel();

        let val = ten + 8;
        unsafe {MUT_INT = 8;};

        zzz(); // #break
        sentinel();

        val as usize
    }];

    zzz(); // #break
    sentinel();
}

fn zzz() {()}
fn sentinel() {()}
