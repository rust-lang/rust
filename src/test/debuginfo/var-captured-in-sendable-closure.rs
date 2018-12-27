// min-lldb-version: 310

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print constant
// gdb-check:$1 = 1
// gdb-command:print a_struct
// gdbg-check:$2 = {a = -2, b = 3.5, c = 4}
// gdbr-check:$2 = var_captured_in_sendable_closure::Struct {a: -2, b: 3.5, c: 4}
// gdb-command:print *owned
// gdb-check:$3 = 5
// gdb-command:continue

// gdb-command:print constant2
// gdb-check:$4 = 6
// gdb-command:continue

// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print constant
// lldbg-check:[...]$0 = 1
// lldbr-check:(isize) constant = 1
// lldb-command:print a_struct
// lldbg-check:[...]$1 = Struct { a: -2, b: 3.5, c: 4 }
// lldbr-check:(var_captured_in_sendable_closure::Struct) a_struct = Struct { a: -2, b: 3.5, c: 4 }
// lldb-command:print *owned
// lldbg-check:[...]$2 = 5
// lldbr-check:(isize) *owned = 5

#![allow(unused_variables)]
#![feature(box_syntax)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

struct Struct {
    a: isize,
    b: f64,
    c: usize
}

fn main() {
    let constant = 1;

    let a_struct = Struct {
        a: -2,
        b: 3.5,
        c: 4
    };

    let owned: Box<_> = box 5;

    let closure = move || {
        zzz(); // #break
        do_something(&constant, &a_struct.a, &*owned);
    };

    closure();

    let constant2 = 6_usize;

    // The `self` argument of the following closure should be passed by value
    // to FnOnce::call_once(self, args), which gets codegened a bit differently
    // than the regular case. Let's make sure this is supported too.
    let immedate_env = move || {
        zzz(); // #break
        return constant2;
    };

    immedate_env();
}

fn do_something(_: &isize, _:&isize, _:&isize) {

}

fn zzz() {()}
