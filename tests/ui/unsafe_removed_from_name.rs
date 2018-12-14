// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unused_imports)]
#![allow(dead_code)]
#![warn(clippy::unsafe_removed_from_name)]

use std::cell::UnsafeCell as TotallySafeCell;

use std::cell::UnsafeCell as TotallySafeCellAgain;

// Shouldn't error
use std::cell::RefCell as ProbablyNotUnsafe;
use std::cell::RefCell as RefCellThatCantBeUnsafe;
use std::cell::UnsafeCell as SuperDangerousUnsafeCell;
use std::cell::UnsafeCell as Dangerunsafe;
use std::cell::UnsafeCell as Bombsawayunsafe;

mod mod_with_some_unsafe_things {
    pub struct Safe {}
    pub struct Unsafe {}
}

use mod_with_some_unsafe_things::Unsafe as LieAboutModSafety;

// Shouldn't error
use mod_with_some_unsafe_things::Safe as IPromiseItsSafeThisTime;
use mod_with_some_unsafe_things::Unsafe as SuperUnsafeModThing;

fn main() {}
