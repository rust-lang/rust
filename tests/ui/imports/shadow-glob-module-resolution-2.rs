// https://github.com/rust-lang/rust/issues/125013

mod a {
  pub mod b {
    pub mod c {
      pub trait D {}
    }
  }
}

use a::*;

use e as b;
//~^ ERROR: unresolved import `e`
use b::c::D as e;
//~^ ERROR: cannot determine resolution for the import
//~| ERROR: cannot determine resolution for the import

fn main() { }
