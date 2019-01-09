// ignore-lldb

// compile-flags:-g
// gdb-command:run

// Test whether compiling a recursive enum definition crashes debug info generation. The test case
// is taken from issue #11083.

#![allow(unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

pub struct Window<'a> {
    callbacks: WindowCallbacks<'a>
}

struct WindowCallbacks<'a> {
    pos_callback: Option<Box<FnMut(&Window, i32, i32) + 'a>>,
}

fn main() {
    let x = WindowCallbacks { pos_callback: None };
}
