// Some versions of the non-rust-enabled LLDB print the wrong generic
// parameter type names in this test.
// rust-lldb

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run

// STACK BY REF
// gdb-command:print *self
// gdbg-check:$1 = {x = {__0 = 8888, __1 = -8888}}
// gdbr-check:$1 = method_on_generic_struct::Struct<(u32, i32)> {x: (8888, -8888)}
// gdb-command:print arg1
// gdb-check:$2 = -1
// gdb-command:print arg2
// gdb-check:$3 = -2
// gdb-command:continue

// STACK BY VAL
// gdb-command:print self
// gdbg-check:$4 = {x = {__0 = 8888, __1 = -8888}}
// gdbr-check:$4 = method_on_generic_struct::Struct<(u32, i32)> {x: (8888, -8888)}
// gdb-command:print arg1
// gdb-check:$5 = -3
// gdb-command:print arg2
// gdb-check:$6 = -4
// gdb-command:continue

// OWNED BY REF
// gdb-command:print *self
// gdbg-check:$7 = {x = 1234.5}
// gdbr-check:$7 = method_on_generic_struct::Struct<f64> {x: 1234.5}
// gdb-command:print arg1
// gdb-check:$8 = -5
// gdb-command:print arg2
// gdb-check:$9 = -6
// gdb-command:continue

// OWNED BY VAL
// gdb-command:print self
// gdbg-check:$10 = {x = 1234.5}
// gdbr-check:$10 = method_on_generic_struct::Struct<f64> {x: 1234.5}
// gdb-command:print arg1
// gdb-check:$11 = -7
// gdb-command:print arg2
// gdb-check:$12 = -8
// gdb-command:continue

// OWNED MOVED
// gdb-command:print *self
// gdbg-check:$13 = {x = 1234.5}
// gdbr-check:$13 = method_on_generic_struct::Struct<f64> {x: 1234.5}
// gdb-command:print arg1
// gdb-check:$14 = -9
// gdb-command:print arg2
// gdb-check:$15 = -10
// gdb-command:continue


// === LLDB TESTS ==================================================================================

// lldb-command:run

// STACK BY REF
// lldb-command:print *self
// lldbg-check:[...]$0 = Struct<(u32, i32)> { x: (8888, -8888) }
// lldbr-check:(method_on_generic_struct::Struct<(u32, i32)>) *self = { x = { = 8888 = -8888 } }
// lldb-command:print arg1
// lldbg-check:[...]$1 = -1
// lldbr-check:(isize) arg1 = -1
// lldb-command:print arg2
// lldbg-check:[...]$2 = -2
// lldbr-check:(isize) arg2 = -2
// lldb-command:continue

// STACK BY VAL
// lldb-command:print self
// lldbg-check:[...]$3 = Struct<(u32, i32)> { x: (8888, -8888) }
// lldbr-check:(method_on_generic_struct::Struct<(u32, i32)>) self = { x = { = 8888 = -8888 } }
// lldb-command:print arg1
// lldbg-check:[...]$4 = -3
// lldbr-check:(isize) arg1 = -3
// lldb-command:print arg2
// lldbg-check:[...]$5 = -4
// lldbr-check:(isize) arg2 = -4
// lldb-command:continue

// OWNED BY REF
// lldb-command:print *self
// lldbg-check:[...]$6 = Struct<f64> { x: 1234.5 }
// lldbr-check:(method_on_generic_struct::Struct<f64>) *self = Struct<f64> { x: 1234.5 }
// lldb-command:print arg1
// lldbg-check:[...]$7 = -5
// lldbr-check:(isize) arg1 = -5
// lldb-command:print arg2
// lldbg-check:[...]$8 = -6
// lldbr-check:(isize) arg2 = -6
// lldb-command:continue

// OWNED BY VAL
// lldb-command:print self
// lldbg-check:[...]$9 = Struct<f64> { x: 1234.5 }
// lldbr-check:(method_on_generic_struct::Struct<f64>) self = Struct<f64> { x: 1234.5 }
// lldb-command:print arg1
// lldbg-check:[...]$10 = -7
// lldbr-check:(isize) arg1 = -7
// lldb-command:print arg2
// lldbg-check:[...]$11 = -8
// lldbr-check:(isize) arg2 = -8
// lldb-command:continue

// OWNED MOVED
// lldb-command:print *self
// lldbg-check:[...]$12 = Struct<f64> { x: 1234.5 }
// lldbr-check:(method_on_generic_struct::Struct<f64>) *self = Struct<f64> { x: 1234.5 }
// lldb-command:print arg1
// lldbg-check:[...]$13 = -9
// lldbr-check:(isize) arg1 = -9
// lldb-command:print arg2
// lldbg-check:[...]$14 = -10
// lldbr-check:(isize) arg2 = -10
// lldb-command:continue

#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

#[derive(Copy, Clone)]
struct Struct<T> {
    x: T
}

impl<T> Struct<T> {

    fn self_by_ref(&self, arg1: isize, arg2: isize) -> isize {
        zzz(); // #break
        arg1 + arg2
    }

    fn self_by_val(self, arg1: isize, arg2: isize) -> isize {
        zzz(); // #break
        arg1 + arg2
    }

    fn self_owned(self: Box<Struct<T>>, arg1: isize, arg2: isize) -> isize {
        zzz(); // #break
        arg1 + arg2
    }
}

fn main() {
    let stack = Struct { x: (8888_u32, -8888_i32) };
    let _ = stack.self_by_ref(-1, -2);
    let _ = stack.self_by_val(-3, -4);

    let owned: Box<_> = Box::new(Struct { x: 1234.5f64 });
    let _ = owned.self_by_ref(-5, -6);
    let _ = owned.self_by_val(-7, -8);
    let _ = owned.self_owned(-9, -10);
}

fn zzz() {()}
