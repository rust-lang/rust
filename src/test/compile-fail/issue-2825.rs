class example {
  let x: int;
  new() { //~ ERROR First constructor declared here
    self.x = 1;
  }
  new(x_: int) {
    self.x = x_;
  }
}

fn main(_args: ~[~str]) {
  let e: example = example();
}
