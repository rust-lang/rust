// aux-build:attr_proc_macro.rs

extern crate attr_proc_macro;
use attr_proc_macro::*;

#[attr_proc_macro] // OK
#[derive(Clone)]
struct Before;

#[derive(Clone)]
#[attr_proc_macro] //~ ERROR macro attributes must be placed before `#[derive]`
struct After;

fn main() {}
