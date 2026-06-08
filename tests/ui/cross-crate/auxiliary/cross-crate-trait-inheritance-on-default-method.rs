//! Auxiliary crate testing this issue https://github.com/rust-lang/rust/issues/3979
#![crate_name="cross_crate_trait_inheritance_on_default_method"]

#![crate_type = "lib"]

pub trait Positioned {
  fn SetX(&mut self, _: isize);
  fn X(&self) -> isize;
}

pub trait Movable: Positioned {
  fn translate(&mut self, dx: isize) {
    let x = self.X() + dx;
    self.SetX(x);
  }
}
