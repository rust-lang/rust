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
    pub struct Safe;
    pub struct Unsafe;
}

use mod_with_some_unsafe_things::Unsafe as LieAboutModSafety;

// merged imports
use mod_with_some_unsafe_things::{Unsafe as A, Unsafe as B};

// Shouldn't error
use mod_with_some_unsafe_things::Safe as IPromiseItsSafeThisTime;

use mod_with_some_unsafe_things::Unsafe as SuperUnsafeModThing;

#[allow(clippy::unsafe_removed_from_name)]
use mod_with_some_unsafe_things::Unsafe as SuperSafeThing;

fn main() {}
