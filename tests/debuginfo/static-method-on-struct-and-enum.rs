// min-lldb-version: 310

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run

// STRUCT
// gdb-command:print arg1
// gdb-check:$1 = 1
// gdb-command:print arg2
// gdb-check:$2 = 2
// gdb-command:continue

// ENUM
// gdb-command:print arg1
// gdb-check:$3 = -3
// gdb-command:print arg2
// gdb-check:$4 = 4.5
// gdb-command:print arg3
// gdb-check:$5 = 5
// gdb-command:continue


// === LLDB TESTS ==================================================================================

// lldb-command:run

// STRUCT
// lldb-command:print arg1
// lldbg-check:[...]$0 = 1
// lldbr-check:(isize) arg1 = 1
// lldb-command:print arg2
// lldbg-check:[...]$1 = 2
// lldbr-check:(isize) arg2 = 2
// lldb-command:continue

// ENUM
// lldb-command:print arg1
// lldbg-check:[...]$2 = -3
// lldbr-check:(isize) arg1 = -3
// lldb-command:print arg2
// lldbg-check:[...]$3 = 4.5
// lldbr-check:(f64) arg2 = 4.5
// lldb-command:print arg3
// lldbg-check:[...]$4 = 5
// lldbr-check:(usize) arg3 = 5
// lldb-command:continue

#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

struct Struct {
    x: isize
}

impl Struct {

    fn static_method(arg1: isize, arg2: isize) -> isize {
        zzz(); // #break
        arg1 + arg2
    }
}

enum Enum {
    Variant1 { x: isize },
    Variant2,
    Variant3(f64, isize, char),
}

impl Enum {

    fn static_method(arg1: isize, arg2: f64, arg3: usize) -> isize {
        zzz(); // #break
        arg1
    }
}

fn main() {
    Struct::static_method(1, 2);
    Enum::static_method(-3, 4.5, 5);
}

fn zzz() {()}
