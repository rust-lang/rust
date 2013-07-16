// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-win32 Broken because of LLVM bug: http://llvm.org/bugs/show_bug.cgi?id=16249

// compile-flags:-Z extra-debug-info
// debugger:break zzz
// debugger:run
// debugger:finish

// debugger:print some
// check:$1 = {0x12345678}

// debugger:print none
// check:$2 = {0x0}

// debugger:print full
// check:$3 = {454545, 0x87654321, 9988}

// debugger:print empty
// check:$4 = {0, 0x0, 0}

// debugger:print droid
// check:$5 = {id = 675675, range = 10000001, internals = 0x43218765}

// debugger:print void_droid
// check:$6 = {id = 0, range = 0, internals = 0x0}


// If a struct has exactly two variants, one of them is empty, and the other one
// contains a non-nullable pointer, then this value is used as the discriminator.
// The test cases in this file make sure that something readable is generated for
// this kind of types.

enum MoreFields<'self> {
    Full(u32, &'self int, i16),
    Empty
}

enum NamedFields<'self> {
    Droid { id: i32, range: i64, internals: &'self int },
    Void
}

fn main() {

    let some: Option<&u32> = Some(unsafe { std::cast::transmute(0x12345678) });
    let none: Option<&u32> = None;

    let full = Full(454545, unsafe { std::cast::transmute(0x87654321) }, 9988);

    let int_val = 0;
    let mut empty = Full(0, &int_val, 0);
    empty = Empty;

    let droid = Droid { id: 675675, range: 10000001, internals: unsafe { std::cast::transmute(0x43218765) } };

    let mut void_droid = Droid { id: 0, range: 0, internals: &int_val };
    void_droid = Void;

    zzz();
}

fn zzz() {()}