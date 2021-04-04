#![crate_type = "lib"]
#![feature(const_generics)]
#![feature(const_generics_defaults)]
#![allow(incomplete_features, dead_code)]

struct Both<const N: usize=3, T> {
//~^ ERROR: generic parameters with a default must be
  v: T
}
