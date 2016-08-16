// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that diverging types default to () (with feature(never_type) disabled).

trait Balls: Sized {
    fn smeg() -> Result<Self, ()>;
}

impl Balls for () {
    fn smeg() -> Result<(), ()> { Ok(()) }
}

struct Flah;

impl Flah {
    fn flah<T: Balls>(&self) -> Result<T, ()> {
        T::smeg()
    }
}

fn doit() -> Result<(), ()> {
    // The type of _ is unconstrained here and should default to ()
    let _ = try!(Flah.flah());
    Ok(())
}

fn main() {
    let _ = doit();
}

