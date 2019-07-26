// run-pass
#![allow(unused_attributes)]
// ignore-windows
// ignore-wasm32-bare no libs to link

#![feature(link_args)]

#[link_args="-lc  -lm"]
#[link_args=" -lc"]
#[link_args="-lc "]
extern {}

fn main() {}
