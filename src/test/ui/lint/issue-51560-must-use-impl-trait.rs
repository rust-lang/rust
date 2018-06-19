// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]
#![deny(unused_must_use)]

trait CompilerHackingToDistractFromHeartbreak {}

#[must_use]
struct IGuessICanMakeNewFriends {
    somehow: bool
}

impl CompilerHackingToDistractFromHeartbreak for IGuessICanMakeNewFriends {}

fn its_not_fair() -> impl CompilerHackingToDistractFromHeartbreak {
    IGuessICanMakeNewFriends { somehow: false }
}

fn main() {
    its_not_fair();
    //~^ ERROR unused `IGuessICanMakeNewFriends` which must be used
}
