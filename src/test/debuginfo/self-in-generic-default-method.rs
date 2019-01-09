// min-lldb-version: 310

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run

// STACK BY REF
// gdb-command:print *self
// gdbg-check:$1 = {x = 987}
// gdbr-check:$1 = self_in_generic_default_method::Struct {x: 987}
// gdb-command:print arg1
// gdb-check:$2 = -1
// gdb-command:print arg2
// gdb-check:$3 = 2
// gdb-command:continue

// STACK BY VAL
// gdb-command:print self
// gdbg-check:$4 = {x = 987}
// gdbr-check:$4 = self_in_generic_default_method::Struct {x: 987}
// gdb-command:print arg1
// gdb-check:$5 = -3
// gdb-command:print arg2
// gdb-check:$6 = -4
// gdb-command:continue

// OWNED BY REF
// gdb-command:print *self
// gdbg-check:$7 = {x = 879}
// gdbr-check:$7 = self_in_generic_default_method::Struct {x: 879}
// gdb-command:print arg1
// gdb-check:$8 = -5
// gdb-command:print arg2
// gdb-check:$9 = -6
// gdb-command:continue

// OWNED BY VAL
// gdb-command:print self
// gdbg-check:$10 = {x = 879}
// gdbr-check:$10 = self_in_generic_default_method::Struct {x: 879}
// gdb-command:print arg1
// gdb-check:$11 = -7
// gdb-command:print arg2
// gdb-check:$12 = -8
// gdb-command:continue

// OWNED MOVED
// gdb-command:print *self
// gdbg-check:$13 = {x = 879}
// gdbr-check:$13 = self_in_generic_default_method::Struct {x: 879}
// gdb-command:print arg1
// gdb-check:$14 = -9
// gdb-command:print arg2
// gdb-check:$15 = -10.5
// gdb-command:continue


// === LLDB TESTS ==================================================================================

// lldb-command:run

// STACK BY REF
// lldb-command:print *self
// lldbg-check:[...]$0 = Struct { x: 987 }
// lldbr-check:(self_in_generic_default_method::Struct) *self = Struct { x: 987 }
// lldb-command:print arg1
// lldbg-check:[...]$1 = -1
// lldbr-check:(isize) arg1 = -1
// lldb-command:print arg2
// lldbg-check:[...]$2 = 2
// lldbr-check:(u16) arg2 = 2
// lldb-command:continue

// STACK BY VAL
// lldb-command:print self
// lldbg-check:[...]$3 = Struct { x: 987 }
// lldbr-check:(self_in_generic_default_method::Struct) self = Struct { x: 987 }
// lldb-command:print arg1
// lldbg-check:[...]$4 = -3
// lldbr-check:(isize) arg1 = -3
// lldb-command:print arg2
// lldbg-check:[...]$5 = -4
// lldbr-check:(i16) arg2 = -4
// lldb-command:continue

// OWNED BY REF
// lldb-command:print *self
// lldbg-check:[...]$6 = Struct { x: 879 }
// lldbr-check:(self_in_generic_default_method::Struct) *self = Struct { x: 879 }
// lldb-command:print arg1
// lldbg-check:[...]$7 = -5
// lldbr-check:(isize) arg1 = -5
// lldb-command:print arg2
// lldbg-check:[...]$8 = -6
// lldbr-check:(i32) arg2 = -6
// lldb-command:continue

// OWNED BY VAL
// lldb-command:print self
// lldbg-check:[...]$9 = Struct { x: 879 }
// lldbr-check:(self_in_generic_default_method::Struct) self = Struct { x: 879 }
// lldb-command:print arg1
// lldbg-check:[...]$10 = -7
// lldbr-check:(isize) arg1 = -7
// lldb-command:print arg2
// lldbg-check:[...]$11 = -8
// lldbr-check:(i64) arg2 = -8
// lldb-command:continue

// OWNED MOVED
// lldb-command:print *self
// lldbg-check:[...]$12 = Struct { x: 879 }
// lldbr-check:(self_in_generic_default_method::Struct) *self = Struct { x: 879 }
// lldb-command:print arg1
// lldbg-check:[...]$13 = -9
// lldbr-check:(isize) arg1 = -9
// lldb-command:print arg2
// lldbg-check:[...]$14 = -10.5
// lldbr-check:(f32) arg2 = -10.5
// lldb-command:continue

#![feature(box_syntax)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

#[derive(Copy, Clone)]
struct Struct {
    x: isize
}

trait Trait : Sized {

    fn self_by_ref<T>(&self, arg1: isize, arg2: T) -> isize {
        zzz(); // #break
        arg1
    }

    fn self_by_val<T>(self, arg1: isize, arg2: T) -> isize {
        zzz(); // #break
        arg1
    }

    fn self_owned<T>(self: Box<Self>, arg1: isize, arg2: T) -> isize {
        zzz(); // #break
        arg1
    }
}

impl Trait for Struct {}

fn main() {
    let stack = Struct { x: 987 };
    let _ = stack.self_by_ref(-1, 2_u16);
    let _ = stack.self_by_val(-3, -4_i16);

    let owned: Box<_> = box Struct { x: 879 };
    let _ = owned.self_by_ref(-5, -6_i32);
    let _ = owned.self_by_val(-7, -8_i64);
    let _ = owned.self_owned(-9, -10.5_f32);
}

fn zzz() {()}
