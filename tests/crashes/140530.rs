//@ known-bug: #140530
//@ edition: 2024

#![feature(async_drop, gen_blocks)]
async gen fn a() {
  _ = async {}
}
fn main() {}
