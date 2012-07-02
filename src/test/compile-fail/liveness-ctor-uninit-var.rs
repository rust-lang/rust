class cat {
  priv {
    let mut meows : uint;
  }

  let how_hungry : int;

  fn eat() {
    self.how_hungry -= 5;
  }

  new(in_x : uint, in_y : int) {
    let foo;
    self.meows = in_x + (in_y as uint);
    self.how_hungry = foo; //~ ERROR use of possibly uninitialized variable: `foo`
  }
}

fn main() {
  let nyan : cat = cat(52u, 99);
  nyan.eat();
}
