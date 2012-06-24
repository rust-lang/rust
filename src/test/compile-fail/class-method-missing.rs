// error-pattern:missing method `eat`
iface animal {
  fn eat();
}

class cat : animal {
  let meows: uint;
  new(in_x : uint) { self.meows = in_x; }
}

fn main() {
  let nyan = cat(0u);
}