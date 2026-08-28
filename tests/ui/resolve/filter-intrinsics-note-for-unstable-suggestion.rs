//@ revisions: gate_off gate_on
//@ aux-build:extern-prelude.rs
//@ dont-require-annotations: NOTE
#![cfg_attr(gate_on, feature(core_io_borrowed_buf))]

extern crate extern_prelude;

fn main() {
  let _: BorrowedBuf;
  //~^ ERROR cannot find type `BorrowedBuf` in this scope
  //[gate_off]~| NOTE 'std::io::BorrowedBuf' is unstable in nightly Rust and is only available with the `#![feature(core_io_borrowed_buf)]` attribute

  let _: HashMap;
  //~^ ERROR cannot find type `HashMap` in this scope

  let _: S;
  //~^ ERROR cannot find type `S` in this scope
}
