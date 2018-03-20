// aux-build:plugin.rs

#[macro_use(WithHelper)]
extern crate plugin;

use plugin::helper;

#[derive(WithHelper)]
#[helper] //~ ERROR `helper` is ambiguous
struct S;

fn main() {}
