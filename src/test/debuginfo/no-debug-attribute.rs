// ignore-lldb

// compile-flags:-g

// gdb-command:run

// gdb-command:info locals
// gdb-check:No locals.
// gdb-command:continue

// gdb-command:info locals
// gdb-check:abc = 10
// gdb-command:continue

#![allow(unused_variables)]
#![feature(no_debug)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

#[inline(never)]
fn id<T>(x: T) -> T {x}

fn function_with_debuginfo() {
    let abc = 10_usize;
    id(abc); // #break
}

#[no_debug]
fn function_without_debuginfo() {
    let abc = -57i32;
    id(abc); // #break
}

fn main() {
    function_without_debuginfo();
    function_with_debuginfo();
}
