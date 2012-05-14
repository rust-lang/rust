class cat {
  let done : native fn(uint);
  let meows : uint;
  new(done: native fn(uint)) {
    self.meows = 0u;
    self.done = done;
  }
  drop { self.done(self.meows); }
}

fn main() {}
