struct cat {
  let done : extern fn(uint);
  let meows : uint;
  new(done: extern fn(uint)) {
    self.meows = 0u;
    self.done = done;
  }
  drop { self.done(self.meows); }
}

fn main() {}
