// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// must-compile-successfully

#![warn(unused)] // UI tests pass `-A unused` (#43896)

struct SoulHistory {
    corridors_of_light: usize,
    hours_are_suns: bool,
    endless_and_singing: bool
}

fn main() {
    let i_think_continually = 2;
    let who_from_the_womb_remembered = SoulHistory {
        corridors_of_light: 5,
        hours_are_suns: true,
        endless_and_singing: true
    };

    if let SoulHistory { corridors_of_light,
                         mut hours_are_suns,
                         endless_and_singing: true } = who_from_the_womb_remembered {
        hours_are_suns = false;
    }
}
