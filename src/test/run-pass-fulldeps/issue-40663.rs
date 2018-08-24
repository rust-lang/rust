// aux-build:custom_derive_plugin.rs
// ignore-stage1

#![feature(plugin, custom_derive)]
#![plugin(custom_derive_plugin)]

#[derive(Nothing, Nothing, Nothing)]
struct S;

fn main() {}
