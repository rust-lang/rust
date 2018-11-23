// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test use of const let without feature gate.

const FOO: usize = {
    //~^ ERROR statements in constants are unstable
    //~| ERROR: let bindings in constants are unstable
    let x = 42;
    //~^ ERROR statements in constants are unstable
    //~| ERROR: let bindings in constants are unstable
    42
};

static BAR: usize = {
    //~^ ERROR statements in statics are unstable
    //~| ERROR: let bindings in statics are unstable
    let x = 42;
    //~^ ERROR statements in statics are unstable
    //~| ERROR: let bindings in statics are unstable
    42
};

fn main() {}
