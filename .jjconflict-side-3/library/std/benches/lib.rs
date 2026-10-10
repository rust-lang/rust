// This is marked as `test = true` and hence picked up by `./x miri`, but that would be too slow.
#![cfg(not(miri))]
#![feature(test)]

extern crate test;

mod hash;
mod path;
mod time;
