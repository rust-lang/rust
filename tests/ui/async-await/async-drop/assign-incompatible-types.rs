// ex-ice: #140530
//@ edition: 2024
//@ build-pass
#![feature(async_drop, gen_blocks)]
#![allow(incomplete_features)]
async gen fn a() {
  _ = async {}
}
fn main() {}
