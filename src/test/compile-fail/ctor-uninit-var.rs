// error-pattern:unsatisfied precondition
class cat {
  priv {
    let mutable meows : uint;
  }

  let how_hungry : int;

  fn eat() {
    how_hungry -= 5;
  }

  new(in_x : uint, in_y : int) {
    let foo;
    meows = in_x + (in_y as uint);
    how_hungry = foo;
  }
}

fn main() {
  let nyan : cat = cat(52u, 99);
  nyan.eat();
}
