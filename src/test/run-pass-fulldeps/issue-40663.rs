#![allow(dead_code)]
// aux-build:custom-derive-plugin.rs
// ignore-stage1

#![feature(plugin)]
#![plugin(custom_derive_plugin)]

#[derive_Nothing]
#[derive_Nothing]
#[derive_Nothing]
struct S;

fn main() {}
