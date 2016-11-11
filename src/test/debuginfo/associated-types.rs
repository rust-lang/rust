// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// min-lldb-version: 310

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
// lldb-check:[...]$0 = Struct<i32> { b: -1, b1: 0 }
// lldb-command:continue

// lldb-command:print inferred
// lldb-check:[...]$1 = 1
// lldb-command:print explicitly
// lldb-check:[...]$2 = 1
// lldb-command:continue

// lldb-command:print arg
// lldb-check:[...]$3 = 2
// lldb-command:continue

// lldb-command:print arg
// lldb-check:[...]$4 = (4, 5)
// lldb-command:continue

// lldb-command:print a
// lldb-check:[...]$5 = 6
// lldb-command:print b
// lldb-check:[...]$6 = 7
// lldb-command:continue

// lldb-command:print a
// lldb-check:[...]$7 = 8
// lldb-command:print b
// lldb-check:[...]$8 = 9
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
