// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test

// compile-flags:-Z extra-debug-info
// debugger:break zzz
// debugger:run
// debugger:finish
// debugger:print simple
// check:$1 = {x = 10, y = 20}

// debugger:print noDestructor
// check:$2 = {a = {x = 10, y = 20}, guard = -1}

// debugger:print withDestructor
// check:$3 = {a = {x = 10, y = 20}, guard = -1}

struct NoDestructor {
    x : i32,
    y : i64
}

struct WithDestructor {
    x : i32,
    y : i64
}

impl Drop for WithDestructor {
    fn finalize(&self) {}
}

struct NoDestructorGuarded {
    a: NoDestructor,
    guard: i64
}

struct WithDestructorGuarded {
    a: WithDestructor,
    guard: i64
}


// The compiler adds a 'destructed' boolean field to structs implementing Drop. This field is used
// at runtime to prevent drop() to be executed more than once (see middle::trans::adt).
// This field must be incorporated by the debug info generation. Otherwise the debugger assumes a
// wrong size/layout for the struct.
fn main() {

    let simple = WithDestructor { x: 10, y: 20 };

    let noDestructor = NoDestructorGuarded {
        a: NoDestructor { x: 10, y: 20 },
        guard: -1
    };

    // If the destructor flag field is not incorporated into the debug info for 'WithDestructor'
    // then the debugger will have an invalid offset for the field 'guard' and thus should not be
    // able to read its value correctly (dots are padding bytes, D is the boolean destructor flag):
    //
    // NoDestructorGuarded = 0000....00000000FFFFFFFF
    //                       <--------------><------>
    //                         NoDestructor   guard
    //
    //
    // withDestructorGuarded = 0000....00000000D.......FFFFFFFF
    //                         <--------------><------>          // How debug info says it is
    //                          WithDestructor  guard
    //
    //                         <----------------------><------>  // How it actually is
    //                              WithDestructor      guard
    //
    let withDestructor = WithDestructorGuarded {
        a: WithDestructor { x: 10, y: 20 },
        guard: -1
    };

    zzz();
}

fn zzz() {()}
