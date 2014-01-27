fn main() {
    let args : ~[~str] = ::std::os::args();
    ::std::io::println(args[0]);
}


// xfail-android: FIXME(#10381)

// This test case checks whether compile unit names are set correctly, so that the correct default
// source file can be found.

// compile-flags:-Z extra-debug-info
// debugger:list
// check:1[...]fn main() {
// check:2[...]let args : ~[~str] = ::std::os::args();
// check:3[...]::std::io::println(args[0]);
// check:4[...]}

// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
