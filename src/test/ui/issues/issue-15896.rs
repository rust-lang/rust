// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #15896. It used to ICE rustc.

fn main() {
    enum R { REB(()) }
    struct Tau { t: usize }
    enum E { B(R, Tau) }

    let e = E::B(R::REB(()), Tau { t: 3 });
    let u = match e {
        E::B(
          Tau{t: x},
          //~^ ERROR mismatched types
          //~| expected type `main::R`
          //~| found type `main::Tau`
          //~| expected enum `main::R`, found struct `main::Tau`
          _) => x,
    };
}
