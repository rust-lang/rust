//! Check that chars can be used in ranges

//@ check-pass

#![feature(pattern_types)]
#![feature(pattern_type_macro)]

use std::pat::pattern_type;

const LOWERCASE: pattern_type!(char is 'a'..='z') = unsafe { std::mem::transmute('b') };

fn main() {}
