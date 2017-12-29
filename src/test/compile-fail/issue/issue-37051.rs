// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_attrs, associated_type_defaults)]
#![allow(warnings)]

trait State: Sized {
    type NextState: State = StateMachineEnded;
    fn execute(self) -> Option<Self::NextState>;
}

struct StateMachineEnded;

impl State for StateMachineEnded {
    fn execute(self) -> Option<Self::NextState> {
        None
    }
}

#[rustc_error]
fn main() { //~ ERROR compilation successful
}
