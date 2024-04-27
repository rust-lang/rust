#![allow(unused)]

#[cfg(all(windows))]
fn hermit() {}

#[cfg(any(windows))]
fn wasi() {}

#[cfg(all(any(unix), all(not(windows))))]
fn the_end() {}

#[cfg(any())]
fn any() {}

fn main() {}
