// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
#![allow(non_upper_case_globals)]

const x: &'static dyn Fn() = &|| println!("ICE here");

fn main() {}
