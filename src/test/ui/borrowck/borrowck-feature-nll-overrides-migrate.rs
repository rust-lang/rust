// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This is a test that the `#![feature(nll)]` opt-in overrides the
// migration mode. The intention here is to emulate the goal behavior
// that `--edition 2018` effects on borrowck (modeled here by `-Z
// borrowck=migrate`) are themselves overridden by the
// `#![feature(nll)]` opt-in.
//
// Therefore, for developer convenience, under `#[feature(nll)]` the
// NLL checks will be emitted as errors *even* in the presence of `-Z
// borrowck=migrate`.

// compile-flags: -Z borrowck=migrate

#![feature(nll)]

fn main() {
    match Some(&4) {
        None => {},
        ref mut foo
            if {
                (|| { let bar = foo; bar.take() })();
                //~^ ERROR cannot move out of borrowed content [E0507]
                false
            } => {},
        Some(ref _s) => println!("Note this arm is bogus; the `Some` became `None` in the guard."),
        _ => println!("Here is some supposedly unreachable code."),
    }
}
