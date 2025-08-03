//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print *a
// gdb-check:$1 = 1
// gdb-command:print *b
// gdb-check:$2 = (2, 3.5)


// === LLDB TESTS ==================================================================================

// lldb-command:run
// lldb-command:v *a
// lldb-check:[...] 1
// lldb-command:v *b
// lldb-check:[...] { 0 = 2 1 = 3.5 }

#![allow(unused_variables)]

fn main() {
    let a = Box::new(1);
    let b = Box::new((2, 3.5f64));

    zzz(); // #break
}

fn zzz() { () }
