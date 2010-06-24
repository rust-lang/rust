// -*- rust -*-

// error-pattern: Infinite type recursion

type x = vec[x];

fn main() {
  let x b = vec();
}
