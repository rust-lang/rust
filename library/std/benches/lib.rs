// Disabling in Miri as these would take too long.
#![cfg(not(miri))]
#![feature(test)]
#![feature(f16)]
#![feature(f128)]

extern crate test;

mod f128;
mod f16;
mod f32;
mod f64;
mod hash;
