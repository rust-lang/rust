// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass

pub struct Stats;

#[derive(PartialEq, Eq)]
pub struct StatVariant {
    pub id: u8,
    _priv: (),
}

#[derive(PartialEq, Eq)]
pub struct Stat {
    pub variant: StatVariant,
    pub index: usize,
    _priv: (),
}

impl Stats {
    pub const TEST: StatVariant = StatVariant{id: 0, _priv: (),};
    #[allow(non_upper_case_globals)]
    pub const A: Stat = Stat{
         variant: Self::TEST,
         index: 0,
         _priv: (),};
}

impl Stat {
    pub fn from_index(variant: StatVariant, index: usize) -> Option<Stat> {
        let stat = Stat{variant, index, _priv: (),};
        match stat {
            Stats::A => Some(Stats::A),
            _ => None,
        }
    }
}

fn main() {}
