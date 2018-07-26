// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum SimpleEnum {
    NoState,
}

struct SimpleStruct {
    no_state_here: u64,
}

fn main() {
    let _ = |simple| {
        match simple {
            SimpleStruct {
                state: 0,
                //~^ struct `SimpleStruct` does not have a field named `state` [E0026]
                ..
            } => (),
        }
    };

    let _ = |simple| {
        match simple {
            SimpleStruct {
                no_state_here: 0,
                no_state_here: 1
                //~^ ERROR field `no_state_here` bound multiple times in the pattern [E0025]
            } => (),
        }
    };

    let _ = |simple| {
        match simple {
            SimpleEnum::NoState {
                state: 0
                //~^ ERROR variant `SimpleEnum::NoState` does not have a field named `state` [E0026]
            } => (),
        }
    };
}
