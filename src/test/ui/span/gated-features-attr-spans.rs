// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(attr_literals)]

#[repr(align(16))] //~ ERROR is experimental
struct Gem {
    mohs_hardness: u8,
    poofed: bool,
    weapon: Weapon,
}

#[repr(simd)] //~ ERROR are experimental
struct Weapon {
    name: String,
    damage: u32
}

impl Gem {
    #[must_use] fn summon_weapon(&self) -> Weapon { self.weapon }
    //~^ WARN is experimental
}

#[must_use] //~ WARN is experimental
fn bubble(gem: Gem) -> Result<Gem, ()> {
    if gem.poofed {
        Ok(gem)
    } else {
        Err(())
    }
}

fn main() {}
