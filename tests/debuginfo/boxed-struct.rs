//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print *boxed_with_padding
// gdb-check:$1 = boxed_struct::StructWithSomePadding {x: 99, y: 999, z: 9999, w: 99999}

// gdb-command:print *boxed_with_dtor
// gdb-check:$2 = boxed_struct::StructWithDestructor {x: 77, y: 777, z: 7777, w: 77777}


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:v *boxed_with_padding
// lldb-check:[...] { x = 99 y = 999 z = 9999 w = 99999 }

// lldb-command:v *boxed_with_dtor
// lldb-check:[...] { x = 77 y = 777 z = 7777 w = 77777 }

#![allow(unused_variables)]

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
