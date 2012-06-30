class cat {
  let how_hungry : int;
  fn meow() {}
  new() {
     self.meow();
     //~^ ERROR use of possibly uninitialized field: `self.how_hungry`
  }
}

fn main() {
}
