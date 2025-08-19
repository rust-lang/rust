//@ min-gdb-version: 14.0
//@ min-lldb-version: 1800

//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================
// gdb-command:run

// gdb-command:print plain_string
// gdb-check:$1 = alloc::string::String {vec: alloc::vec::Vec<u8, alloc::alloc::Global> {buf: alloc::raw_vec::RawVec<u8, alloc::alloc::Global> {inner: alloc::raw_vec::RawVecInner<alloc::alloc::Global> {ptr: core::ptr::unique::Unique<u8> {pointer: core::ptr::non_null::NonNull<u8> {pointer: 0x[...]}, _marker: core::marker::PhantomData<u8>}, cap: core::num::niche_types::UsizeNoHighBit (5), alloc: alloc::alloc::Global}, _marker: core::marker::PhantomData<u8>}, len: 5}}

// gdb-command:print plain_str
// gdb-check:$2 = "Hello"

// gdb-command:print str_in_struct
// gdb-check:$3 = strings_and_strs::Foo {inner: "Hello"}

// gdb-command:print str_in_tuple
// gdb-check:$4 = ("Hello", "World")

// gdb-command:print str_in_rc
// gdb-check:$5 = alloc::rc::Rc<&str, alloc::alloc::Global> {ptr: core::ptr::non_null::NonNull<alloc::rc::RcInner<&str>> {pointer: 0x[...]}, phantom: core::marker::PhantomData<alloc::rc::RcInner<&str>>, alloc: alloc::alloc::Global}

// === LLDB TESTS ==================================================================================
// lldb-command:run
// lldb-command:v plain_string
// lldb-check:(alloc::string::String) plain_string = "Hello" { [0] = 'H' [1] = 'e' [2] = 'l' [3] = 'l' [4] = 'o' }

// lldb-command:v plain_str
// lldb-check:(&str) plain_str = "Hello" { [0] = 'H' [1] = 'e' [2] = 'l' [3] = 'l' [4] = 'o' }

// lldb-command:v str_in_struct
// lldb-check:((&str, &str)) str_in_tuple = { 0 = "Hello" { [0] = 'H' [1] = 'e' [2] = 'l' [3] = 'l' [4] = 'o' } 1 = "World" { [0] = 'W' [1] = 'o' [2] = 'r' [3] = 'l' [4] = 'd' } }

// lldb-command:v str_in_tuple
// lldb-check:((&str, &str)) str_in_tuple = { 0 = "Hello" { [0] = 'H' [1] = 'e' [2] = 'l' [3] = 'l' [4] = 'o' } 1 = "World" { [0] = 'W' [1] = 'o' [2] = 'r' [3] = 'l' [4] = 'd' } }

// lldb-command:v str_in_rc
// lldb-check:(alloc::rc::Rc<&str, alloc::alloc::Global>) str_in_rc = strong=1, weak=0 { value = "Hello" { [0] = 'H' [1] = 'e' [2] = 'l' [3] = 'l' [4] = 'o' } }


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
