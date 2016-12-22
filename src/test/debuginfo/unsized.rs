// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print *a
// gdbg-check:$1 = {value = [...] "abc"}
// gdbr-check:$1 = unsized::Foo<[u8]> {value: [...]}

// gdb-command:print *b
// gdbg-check:$2 = {value = {value = [...] "abc"}}
// gdbr-check:$2 = unsized::Foo<unsized::Foo<[u8]>> {value: unsized::Foo<[u8]> {value: [...]}}


#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

struct Foo<T: ?Sized> {
    value: T
}

fn main() {
    let foo: Foo<Foo<[u8; 4]>> = Foo {
        value: Foo {
            value: *b"abc\0"
        }
    };
    let a: &Foo<[u8]> = &foo.value;
    let b: &Foo<Foo<[u8]>> = &foo;

    zzz(); // #break
}

fn zzz() { () }
