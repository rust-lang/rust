// Test that `objc::class!` and `objc::selector!` aren't `const` expressions.
// The system gives them their final values at dynamic load time.

//@ edition: 2024
//@ only-apple

#![feature(darwin_objc)]

use std::os::darwin::objc;

pub const CLASS: objc::Class = objc::class!("NSObject");
//~^ ERROR cannot access extern static `CLASS::VAL` [E0080]

pub const SELECTOR: objc::SEL = objc::selector!("alloc");
//~^ ERROR cannot access extern static `SELECTOR::VAL` [E0080]

pub fn main() {}
