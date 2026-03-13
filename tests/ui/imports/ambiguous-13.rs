// https://github.com/rust-lang/rust/pull/113099#issuecomment-1637022296

pub mod object {
    #[derive(Debug)]
    pub struct Rect;
}

pub mod content {
  use crate::object::*;

  #[derive(Debug)]
  pub struct Rect;
}

use crate::object::*;
use crate::content::*;

fn a(_: Rect) {}
//~^ ERROR `Rect` is ambiguous
//~| WARNING this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
fn main() { }
