//@ ignore-gdb
//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// Check that structs get placed in the correct namespace

// lldb-command:run
// lldb-command:v struct1
// lldb-check:(struct_namespace::Struct1)[...]
// lldb-command:v struct2
// lldb-check:(struct_namespace::Struct2)[...]

// lldb-command:v mod1_struct1
// lldb-check:(struct_namespace::mod1::Struct1)[...]
// lldb-command:v mod1_struct2
// lldb-check:(struct_namespace::mod1::Struct2)[...]

#![allow(unused_variables)]
#![allow(dead_code)]

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
