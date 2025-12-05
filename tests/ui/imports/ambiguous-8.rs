//@ aux-build: ambiguous-8-extern.rs

extern crate ambiguous_8_extern;

mod s {
  pub trait Error {}
}

use s::*;
use ambiguous_8_extern::*;
fn a<E: Error>(_: E) {}
//~^ ERROR `Error` is ambiguous

fn main() {}
