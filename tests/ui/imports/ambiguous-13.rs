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

fn main() { }
