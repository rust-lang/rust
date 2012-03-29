// error-pattern:assigning to immutable field
class cat {
  priv {
    let mut meows : uint;
  }

  let how_hungry : int;

  fn speak() { self.meows += 1u; }
  new(in_x : uint, in_y : int) { self.meows = in_x; self.how_hungry = in_y; }
}

fn main() {
  let nyan : cat = cat(52u, 99);
  nyan.speak = fn@() { log(error, "meow"); };
}
