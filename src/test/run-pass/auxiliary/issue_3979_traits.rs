#![crate_name="issue_3979_traits"]

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
