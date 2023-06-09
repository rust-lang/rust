// error-pattern:nonexistent
class cat : nonexistent {
  let meows: usize;
  new(in_x : usize) { self.meows = in_x; }
}

fn main() {
  let nyan = cat(0);
}
