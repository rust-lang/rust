// error-pattern:is not compiled with this crate's panic strategy `abort`
// compile-flags:-C panic=abort
// ignore-wasm32-bare compiled with panic=abort by default

#![feature(test)]

extern crate test;

fn main() {
}
