// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_snake_case)]

register_diagnostic! { E0001, r##"
    This error suggests that the expression arm corresponding to the noted pattern
    will never be reached as for all possible values of the expression being matched,
    one of the preceeding patterns will match.

    This means that perhaps some of the preceeding patterns are too general, this
    one is too specific or the ordering is incorrect.
"## }

register_diagnostics! {
    E0002,
    E0003,
    E0004,
    E0005,
    E0006,
    E0007,
    E0008,
    E0009,
    E0010,
    E0011,
    E0012,
    E0013,
    E0014,
    E0015,
    E0016,
    E0017,
    E0018,
    E0019,
    E0020,
    E0022,
    E0109,
    E0110,
    E0133,
    E0134,
    E0135,
    E0136,
    E0137,
    E0138,
    E0139,
    E0140,
    E0152,
    E0153,
    E0157,
    E0158,
    E0161,
    E0162,
    E0165,
    E0166,
    E0167,
    E0168,
    E0169,
    E0170,
    E0171,
    E0172,
    E0173,
    E0174,
    E0177,
    E0178
}
