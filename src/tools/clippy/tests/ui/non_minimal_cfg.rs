#![allow(unused)]

#[cfg(all(windows))]
//~^ non_minimal_cfg
fn hermit() {}

#[cfg(any(windows))]
//~^ non_minimal_cfg
fn wasi() {}

#[cfg(all(any(unix), all(not(windows))))]
//~^ non_minimal_cfg
//~| non_minimal_cfg
fn the_end() {}

#[cfg(any())]
fn any() {}

fn main() {}
