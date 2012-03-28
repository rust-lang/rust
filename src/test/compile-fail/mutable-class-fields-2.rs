// error-pattern:assigning to immutable class field
class cat {
  priv {
    let mutable meows : uint;
  }

  let how_hungry : int;

  fn eat() {
    how_hungry -= 5;
  }

  new(in_x : uint, in_y : int) { meows = in_x; how_hungry = in_y; }
}

fn main() {
  let nyan : cat = cat(52u, 99);
  nyan.eat();
}
