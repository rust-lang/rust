// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// ignore-lldb
// ignore-gdb // Test temporarily ignored due to debuginfo tests being disabled, see PR 47155

// compile-flags:-C debuginfo=1

// Make sure functions have proper names
// gdb-command:info functions
// gdbg-check:[...]void[...]main([...]);
// gdbr-check:fn limited_debuginfo::main();
// gdbg-check:[...]void[...]some_function([...]);
// gdbr-check:fn limited_debuginfo::some_function();
// gdbg-check:[...]void[...]some_other_function([...]);
// gdbr-check:fn limited_debuginfo::some_other_function();
// gdbg-check:[...]void[...]zzz([...]);
// gdbr-check:fn limited_debuginfo::zzz();

// gdb-command:run

// Make sure there is no information about locals
// gdb-command:info locals
// gdb-check:No locals.
// gdb-command:continue


#![allow(unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

struct Struct {
    a: i64,
    b: i32
}

fn main() {
    some_function(101, 202);
    some_other_function(1, 2);
}


fn zzz() {()}

fn some_function(a: isize, b: isize) {
    let some_variable = Struct { a: 11, b: 22 };
    let some_other_variable = 23;

    for x in 0..1 {
        zzz(); // #break
    }
}

fn some_other_function(a: isize, b: isize) -> bool { true }
