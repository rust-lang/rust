class cat : nonexistent { //~ ERROR expected one of `!` or `::`, found `cat`
  let meows: usize;
  new(in_x : usize) { self.meows = in_x; }
}

fn main() {
  let nyan = cat(0);
}
