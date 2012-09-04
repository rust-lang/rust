// error-pattern:missing method `eat`
trait animal {
  fn eat();
}

struct cat : animal {
  let meows: uint;
  new(in_x : uint) { self.meows = in_x; }
}

fn main() {
  let nyan = cat(0u);
}