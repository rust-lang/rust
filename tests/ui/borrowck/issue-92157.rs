//@ add-core-stubs
#![feature(no_core)]
#![feature(lang_items)]

#![no_core]

#[cfg(target_os = "linux")]
#[link(name = "c")]
extern "C" {}

#[lang = "start"]
fn start<T>(_main: fn() -> T, _argc: isize, _argv: *const *const u8) -> isize {
    //~^ ERROR lang item `start` function has wrong type [E0308]
    40+2
}

extern crate minicore;
use minicore::*;

fn main() {}
