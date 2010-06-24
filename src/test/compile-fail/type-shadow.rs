// -*- rust -*-

// error-pattern: mismatched types

fn main() {
  type X = int;
  type Y = X;
  if (true) {
    type X = str;
    let Y y = "hello";
  }
}
