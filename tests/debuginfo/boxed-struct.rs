// min-lldb-version: 310

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print *boxed_with_padding
// gdbg-check:$1 = {x = 99, y = 999, z = 9999, w = 99999}
// gdbr-check:$1 = boxed_struct::StructWithSomePadding {x: 99, y: 999, z: 9999, w: 99999}

// gdb-command:print *boxed_with_dtor
// gdbg-check:$2 = {x = 77, y = 777, z = 7777, w = 77777}
// gdbr-check:$2 = boxed_struct::StructWithDestructor {x: 77, y: 777, z: 7777, w: 77777}


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print *boxed_with_padding
// lldbg-check:[...]$0 = { x = 99 y = 999 z = 9999 w = 99999 }
// lldbr-check:(boxed_struct::StructWithSomePadding) *boxed_with_padding = { x = 99 y = 999 z = 9999 w = 99999 }

// lldb-command:print *boxed_with_dtor
// lldbg-check:[...]$1 = { x = 77 y = 777 z = 7777 w = 77777 }
// lldbr-check:(boxed_struct::StructWithDestructor) *boxed_with_dtor = { x = 77 y = 777 z = 7777 w = 77777 }

#![allow(unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

struct StructWithSomePadding {
    x: i16,
    y: i32,
    z: i32,
    w: i64
}

struct StructWithDestructor {
    x: i16,
    y: i32,
    z: i32,
    w: i64
}

impl Drop for StructWithDestructor {
    fn drop(&mut self) {}
}

fn main() {

    let boxed_with_padding: Box<_> = Box::new(StructWithSomePadding {
        x: 99,
        y: 999,
        z: 9999,
        w: 99999,
    });

    let boxed_with_dtor: Box<_> = Box::new(StructWithDestructor {
        x: 77,
        y: 777,
        z: 7777,
        w: 77777,
    });
    zzz(); // #break
}

fn zzz() { () }
