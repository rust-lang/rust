// ignore-gdb
// compile-flags:-g
// min-lldb-version: 310

// Check that structs get placed in the correct namespace

// lldb-command:run
// lldb-command:p struct1
// lldbg-check:(struct_namespace::Struct1) $0 = [...]
// lldbr-check:(struct_namespace::Struct1) struct1 = Struct1 { a: 0, b: 1 }
// lldb-command:p struct2
// lldbg-check:(struct_namespace::Struct2) $1 = [...]
// lldbr-check:(struct_namespace::Struct2) struct2 = { = 2 }

// lldb-command:p mod1_struct1
// lldbg-check:(struct_namespace::mod1::Struct1) $2 = [...]
// lldbr-check:(struct_namespace::mod1::Struct1) mod1_struct1 = Struct1 { a: 3, b: 4 }
// lldb-command:p mod1_struct2
// lldbg-check:(struct_namespace::mod1::Struct2) $3 = [...]
// lldbr-check:(struct_namespace::mod1::Struct2) mod1_struct2 = { = 5 }

#![allow(unused_variables)]
#![allow(dead_code)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

struct Struct1 {
    a: u32,
    b: u64,
}

struct Struct2(u32);

mod mod1 {

    pub struct Struct1 {
        pub a: u32,
        pub b: u64,
    }

    pub struct Struct2(pub u32);
}


fn main() {
    let struct1 = Struct1 {
        a: 0,
        b: 1,
    };

    let struct2 = Struct2(2);

    let mod1_struct1 = mod1::Struct1 {
        a: 3,
        b: 4,
    };

    let mod1_struct2 = mod1::Struct2(5);

    zzz(); // #break
}

#[inline(never)]
fn zzz() {()}
