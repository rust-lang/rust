// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This is testing an attempt to corrupt the discriminant of the match
// arm in a guard, followed by an attempt to continue matching on that
// corrupted discriminant in the remaining match arms.
//
// Basically this is testing that our new NLL feature of emitting a
// fake read on each match arm is catching cases like this.
//
// This case is interesting because it includes a guard that
// diverges, and therefore a single final fake-read at the very end
// after the final match arm would not suffice.

#![feature(nll)]

struct ForceFnOnce;

fn main() {
    let mut x = &mut Some(&2);
    let force_fn_once = ForceFnOnce;
    match x {
        &mut None => panic!("unreachable"),
        &mut Some(&_) if {
            // ForceFnOnce needed to exploit #27282
            (|| { *x = None; drop(force_fn_once); })();
            //~^ ERROR closure requires unique access to `x` but it is already borrowed [E0500]
            false
        } => {}
        &mut Some(&a) if { // this binds to garbage if we've corrupted discriminant
            println!("{}", a);
            panic!()
        } => {}
        _ => panic!("unreachable"),
    }
}
