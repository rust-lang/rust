// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:-Z extra-debug-info
// debugger:set print union on
// debugger:break zzz
// debugger:run
// debugger:finish

// d ebugger:print case2
// c heck:$1 = {Case1, 0, 1}

// debugger:print univariant
// check:$1 = {{-1}}


enum Regular {
    Case1(i32, i64),
    Case2(bool, i16, i32)
}

enum Univariant {
	TheOnlyCase(i64)
}

fn main() {

    let case1 = Case1(110, 220);
    let case2 = Case2(false, 2, 3);

    let univariant = TheOnlyCase(-1);

    zzz();
}

fn zzz() {()}