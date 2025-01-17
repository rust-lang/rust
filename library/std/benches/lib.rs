// Disabling in Miri as these would take too long.
#![cfg(not(miri))]
#![feature(test)]

extern crate test;

mod hash;
mod path;
