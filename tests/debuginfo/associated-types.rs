//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================
// gdb-command:run

// gdb-command:print arg
// gdb-check:$1 = associated_types::Struct<i32> {b: -1, b1: 0}
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
// gdb-check:$5 = (4, 5)
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

// lldb-command:v arg
// lldb-check:[...] { b = -1 b1 = 0 }
// lldb-command:continue

// lldb-command:v inferred
// lldb-check:[...] 1
// lldb-command:v explicitly
// lldb-check:[...] 1
// lldb-command:continue

// lldb-command:v arg
// lldb-check:[...] 2
// lldb-command:continue

// lldb-command:v arg
// lldb-check:[...] { 0 = 4 1 = 5 }
// lldb-command:continue

// lldb-command:v a
// lldb-check:[...] 6
// lldb-command:v b
// lldb-check:[...] 7
// lldb-command:continue

// lldb-command:v a
// lldb-check:[...] 8
// lldb-command:v b
// lldb-check:[...] 9
// lldb-command:continue

#![allow(unused_variables)]
#![allow(dead_code)]

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
