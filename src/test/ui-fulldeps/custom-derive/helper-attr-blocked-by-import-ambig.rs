// aux-build:plugin.rs
// ignore-stage1

#[macro_use(WithHelper)]
extern crate plugin;

use plugin::helper;

#[derive(WithHelper)]
#[helper] //~ ERROR `helper` is ambiguous
struct S;

fn main() {}
