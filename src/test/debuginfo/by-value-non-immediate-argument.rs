// ignore-tidy-linelength
// ignore-test // Test temporarily ignored due to debuginfo tests being disabled, see PR 47155
// min-lldb-version: 310

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print s
// gdbg-check:$1 = {a = 1, b = 2.5}
// gdbr-check:$1 = by_value_non_immediate_argument::Struct {a: 1, b: 2.5}
// gdb-command:continue

// gdb-command:print x
// gdbg-check:$2 = {a = 3, b = 4.5}
// gdbr-check:$2 = by_value_non_immediate_argument::Struct {a: 3, b: 4.5}
// gdb-command:print y
// gdb-check:$3 = 5
// gdb-command:print z
// gdb-check:$4 = 6.5
// gdb-command:continue

// gdb-command:print a
// gdbg-check:$5 = {__0 = 7, __1 = 8, __2 = 9.5, __3 = 10.5}
// gdbr-check:$5 = (7, 8, 9.5, 10.5)
// gdb-command:continue

// gdb-command:print a
// gdbg-check:$6 = {__0 = 11.5, __1 = 12.5, __2 = 13, __3 = 14}
// gdbr-check:$6 = by_value_non_immediate_argument::Newtype (11.5, 12.5, 13, 14)
// gdb-command:continue

// gdb-command:print x
// gdbg-check:$7 = {{RUST$ENUM$DISR = Case1, x = 0, y = 8970181431921507452}, {RUST$ENUM$DISR = Case1, [...]}}
// gdbr-check:$7 = by_value_non_immediate_argument::Enum::Case1{x: 0, y: 8970181431921507452}
// gdb-command:continue


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print s
// lldb-check:[...]$0 = Struct { a: 1, b: 2.5 }
// lldb-command:continue

// lldb-command:print x
// lldb-check:[...]$1 = Struct { a: 3, b: 4.5 }
// lldb-command:print y
// lldb-check:[...]$2 = 5
// lldb-command:print z
// lldb-check:[...]$3 = 6.5
// lldb-command:continue

// lldb-command:print a
// lldb-check:[...]$4 = (7, 8, 9.5, 10.5)
// lldb-command:continue

// lldb-command:print a
// lldb-check:[...]$5 = Newtype(11.5, 12.5, 13, 14)
// lldb-command:continue

// lldb-command:print x
// lldb-check:[...]$6 = Case1 { x: 0, y: 8970181431921507452 }
// lldb-command:continue

#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

#[derive(Clone)]
struct Struct {
    a: isize,
    b: f64
}

#[derive(Clone)]
struct StructStruct {
    a: Struct,
    b: Struct
}

fn fun(s: Struct) {
    zzz(); // #break
}

fn fun_fun(StructStruct { a: x, b: Struct { a: y, b: z } }: StructStruct) {
    zzz(); // #break
}

fn tup(a: (isize, usize, f64, f64)) {
    zzz(); // #break
}

struct Newtype(f64, f64, isize, usize);

fn new_type(a: Newtype) {
    zzz(); // #break
}

// The first element is to ensure proper alignment, irrespective of the machines word size. Since
// the size of the discriminant value is machine dependent, this has be taken into account when
// datatype layout should be predictable as in this case.
enum Enum {
    Case1 { x: i64, y: i64 },
    Case2 (i64, i32, i32),
}

fn by_val_enum(x: Enum) {
    zzz(); // #break
}

fn main() {
    fun(Struct { a: 1, b: 2.5 });
    fun_fun(StructStruct { a: Struct { a: 3, b: 4.5 }, b: Struct { a: 5, b: 6.5 } });
    tup((7, 8, 9.5, 10.5));
    new_type(Newtype(11.5, 12.5, 13, 14));

    // 0b0111110001111100011111000111110001111100011111000111110001111100 = 8970181431921507452
    // 0b01111100011111000111110001111100 = 2088533116
    // 0b0111110001111100 = 31868
    // 0b01111100 = 124
    by_val_enum(Enum::Case1 { x: 0, y: 8970181431921507452 });
}

fn zzz() { () }
