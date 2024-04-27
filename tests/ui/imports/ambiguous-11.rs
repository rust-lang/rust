//@ aux-build: ambiguous-11-extern.rs

extern crate ambiguous_11_extern;

mod s {
  pub trait Error {}
}

use s::*;
use ambiguous_11_extern::*;
fn a<E: Error>(_: E) {}
//~^ ERROR `Error` is ambiguous

fn main() {}
