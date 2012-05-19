pure fn is_even(i: int) -> bool { (i%2) == 0 }
fn even(i: int) : is_even(i) -> int { i }

class cat {
  priv {
    let mut meows : uint;
  }

  let how_hungry : int;

  fn eat() {
    self.how_hungry -= 5;
  }

  new(in_x : uint, in_y : int) {
    let foo = 3;
    self.meows = in_x + (in_y as uint);
    self.how_hungry = even(foo); //! ERROR unsatisfied precondition
  }
}

fn main() {
  let nyan : cat = cat(52u, 99);
  nyan.eat();
}
