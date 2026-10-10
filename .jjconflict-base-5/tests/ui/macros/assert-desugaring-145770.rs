//! Regression test for #145770.
//!
//! Changing the `assert!` desugaring from an `if !cond {}` to `match` expression is
//! backwards-incompatible, and may need to be done over an edition boundary or limit editions for
//! which the desguaring change impacts.

//@ check-pass

#[derive(Debug)]
struct F {
    data: bool
}

impl std::ops::Not for F {
  type Output = bool;
  fn not(self) -> Self::Output { !self.data }
}

fn main() {
  let f = F { data: true };

  assert!(f);
}
