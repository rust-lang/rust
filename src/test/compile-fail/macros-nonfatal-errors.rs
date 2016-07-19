// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// test that errors in a (selection) of macros don't kill compilation
// immediately, so that we get more errors listed at a time.

#![feature(asm)]
#![feature(trace_macros, concat_idents)]

#[derive(Default, //~ ERROR
           Zero)] //~ ERROR
enum CantDeriveThose {}

fn main() {
    doesnt_exist!(); //~ ERROR

    asm!(invalid); //~ ERROR

    concat_idents!("not", "idents"); //~ ERROR

    option_env!(invalid); //~ ERROR
    env!(invalid); //~ ERROR
    env!(foo, abr, baz); //~ ERROR
    env!("RUST_HOPEFULLY_THIS_DOESNT_EXIST"); //~ ERROR

    foo::blah!(); //~ ERROR

    format!(invalid); //~ ERROR

    include!(invalid); //~ ERROR

    include_str!(invalid); //~ ERROR
    include_str!("i'd be quite surprised if a file with this name existed"); //~ ERROR
    include_bytes!(invalid); //~ ERROR
    include_bytes!("i'd be quite surprised if a file with this name existed"); //~ ERROR

    trace_macros!(invalid); //~ ERROR
}
