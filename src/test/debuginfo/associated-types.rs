// Some versions of the non-rust-enabled LLDB print the wrong generic
// parameter type names in this test.
// rust-lldb

// compile-flags:-g

// === GDB TESTS ===================================================================================
// gdb-command:run

// gdb-command:print arg
// gdbg-check:$1 = {b = -1, b1 = 0}
// gdbr-check:$1 = associated_types::Struct<i32> {b: -1, b1: 0}
// gdb-command:continue

// gdb-command:print inferred
// gdb-check:$2 = 1
// gdb-command:print explicitly
// gdb-check:$3 = 1
// gdb-command:continue

// gdb-command:print arg
// gdb-check:$4 = 2
// gdb-command:continue

// gdb-command:print arg
// gdbg-check:$5 = {__0 = 4, __1 = 5}
// gdbr-check:$5 = (4, 5)
// gdb-command:continue

// gdb-command:print a
// gdb-check:$6 = 6
// gdb-command:print b
// gdb-check:$7 = 7
// gdb-command:continue

// gdb-command:print a
// gdb-check:$8 = 8
// gdb-command:print b
// gdb-check:$9 = 9
// gdb-command:continue

// === LLDB TESTS ==================================================================================
// lldb-command:run

// lldb-command:print arg
// lldbg-check:[...]$0 = Struct<i32> { b: -1, b1: 0 }
// lldbr-check:(associated_types::Struct<i32>) arg = Struct<i32> { b: -1, b1: 0 }
// lldb-command:continue

// lldb-command:print inferred
// lldbg-check:[...]$1 = 1
// lldbr-check:(i64) inferred = 1
// lldb-command:print explicitly
// lldbg-check:[...]$2 = 1
// lldbr-check:(i64) explicitly = 1
// lldb-command:continue

// lldb-command:print arg
// lldbg-check:[...]$3 = 2
// lldbr-check:(i64) arg = 2
// lldb-command:continue

// lldb-command:print arg
// lldbg-check:[...]$4 = (4, 5)
// lldbr-check:((i32, i64)) arg = { = 4 = 5 }
// lldb-command:continue

// lldb-command:print a
// lldbg-check:[...]$5 = 6
// lldbr-check:(i32) a = 6
// lldb-command:print b
// lldbg-check:[...]$6 = 7
// lldbr-check:(i64) b = 7
// lldb-command:continue

// lldb-command:print a
// lldbg-check:[...]$7 = 8
// lldbr-check:(i64) a = 8
// lldb-command:print b
// lldbg-check:[...]$8 = 9
// lldbr-check:(i32) b = 9
// lldb-command:continue

#![allow(unused_variables)]
#![allow(dead_code)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

trait TraitWithAssocType {
    type Type;

    fn get_value(&self) -> Self::Type;
}
impl TraitWithAssocType for i32 {
    type Type = i64;

    fn get_value(&self) -> i64 { *self as i64 }
}

struct Struct<T: TraitWithAssocType> {
    b: T,
    b1: T::Type,
}

enum Enum<T: TraitWithAssocType> {
    Variant1(T, T::Type),
    Variant2(T::Type, T)
}

fn assoc_struct<T: TraitWithAssocType>(arg: Struct<T>) {
    zzz(); // #break
}

fn assoc_local<T: TraitWithAssocType>(x: T) {
    let inferred = x.get_value();
    let explicitly: T::Type = x.get_value();

    zzz(); // #break
}

fn assoc_arg<T: TraitWithAssocType>(arg: T::Type) {
    zzz(); // #break
}

fn assoc_return_value<T: TraitWithAssocType>(arg: T) -> T::Type {
    return arg.get_value();
}

fn assoc_tuple<T: TraitWithAssocType>(arg: (T, T::Type)) {
    zzz(); // #break
}

fn assoc_enum<T: TraitWithAssocType>(arg: Enum<T>) {

    match arg {
        Enum::Variant1(a, b) => {
            zzz(); // #break
        }
        Enum::Variant2(a, b) => {
            zzz(); // #break
        }
    }
}

fn main() {
    assoc_struct(Struct { b: -1, b1: 0 });
    assoc_local(1);
    assoc_arg::<i32>(2);
    assoc_return_value(3);
    assoc_tuple((4, 5));
    assoc_enum(Enum::Variant1(6, 7));
    assoc_enum(Enum::Variant2(8, 9));
}

fn zzz() { () }
