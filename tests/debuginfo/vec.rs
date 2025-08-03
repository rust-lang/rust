//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:run
// gdb-command:print a
// gdb-check:$1 = [1, 2, 3]
// gdb-command:print vec::VECT
// gdb-check:$2 = [4, 5, 6]


// === LLDB TESTS ==================================================================================

// lldb-command:run
// lldb-command:v a
// lldb-check:[...] { [0] = 1 [1] = 2 [2] = 3 }

#![allow(unused_variables)]

static mut VECT: [i32; 3] = [1, 2, 3];

fn main() {
    let a = [1, 2, 3];

    unsafe {
        VECT[0] = 4;
        VECT[1] = 5;
        VECT[2] = 6;
    }

    zzz(); // #break
}

fn zzz() {()}
