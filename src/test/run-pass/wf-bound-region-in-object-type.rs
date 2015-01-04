// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the `wf` checker properly handles bound regions in object
// types. Compiling this code used to trigger an ICE.

pub struct Context<'tcx> {
    vec: &'tcx Vec<int>
}

pub type Cmd<'a> = &'a int;

pub type DecodeInlinedItem<'a> =
    Box<for<'tcx> FnMut(Cmd, &Context<'tcx>) -> Result<&'tcx int, ()> + 'a>;

fn foo(d: DecodeInlinedItem) {
}

fn main() { }
