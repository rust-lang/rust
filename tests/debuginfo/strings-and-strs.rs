// Require a gdb or lldb that can read DW_TAG_variant_part.
//@ min-gdb-version: 8.2
//@ min-lldb-version: 1800

//@ compile-flags:-g

// === GDB TESTS ===================================================================================
// gdb-command:run

// gdb-command:print plain_string
// gdbr-check:$1 = "Hello"

// gdb-command:print plain_str
// gdbr-check:$2 = "Hello"

// gdb-command:print str_in_struct
// gdbr-check:$3 = strings_and_strs::Foo {inner: "Hello"}

// gdb-command:print str_in_tuple
// gdbr-check:$4 = ("Hello", "World")

// gdb-command:print str_in_rc
// gdbr-check:$5 = Rc(strong=1, weak=0) = {value = "Hello", strong = 1, weak = 0}


// === LLDB TESTS ==================================================================================
// lldb-command:run
// lldb-command:v plain_string
// lldbg-check:(alloc::string::String) plain_string = "Hello" { vec = size=5 { [0] = 'H' [1] = 'e' [2] = 'l' [3] = 'l' [4] = 'o' } }

// lldb-command:v plain_str
// lldbg-check:(&str) plain_str = "Hello" { [0] = 'H' [1] = 'e' [2] = 'l' [3] = 'l' [4] = 'o' }

// lldb-command:v str_in_struct
// lldbg-check:((&str, &str)) str_in_tuple = { 0 = "Hello" { [0] = 'H' [1] = 'e' [2] = 'l' [3] = 'l' [4] = 'o' } 1 = "World" { [0] = 'W' [1] = 'o' [2] = 'r' [3] = 'l' [4] = 'd' } }

// lldb-command:v str_in_tuple
// lldbg-check:((&str, &str)) str_in_tuple = { 0 = "Hello" { [0] = 'H' [1] = 'e' [2] = 'l' [3] = 'l' [4] = 'o' } 1 = "World" { [0] = 'W' [1] = 'o' [2] = 'r' [3] = 'l' [4] = 'd' } }

// lldb-command:v str_in_rc
// lldbg-check:(alloc::rc::Rc<&str, alloc::alloc::Global>) str_in_rc = strong=1, weak=0 { value = "Hello" { [0] = 'H' [1] = 'e' [2] = 'l' [3] = 'l' [4] = 'o' } }


#![allow(unused_variables)]

pub struct Foo<'a> {
    inner: &'a str,
}

fn main() {
    let plain_string = String::from("Hello");
    let plain_str = "Hello";
    let str_in_struct = Foo { inner: "Hello" };
    let str_in_tuple = ("Hello", "World");

    let str_in_rc = std::rc::Rc::new("Hello");
    zzz(); // #break
}

fn zzz() {
    ()
}
