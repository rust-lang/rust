//@ pp-exact
//@ edition: 2024

#![feature(try_blocks, try_blocks_heterogeneous)]

fn main() { try { Some(1)? }; try bikeshed Result<u32, ()> { 3 }; }
