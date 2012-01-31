// -*- rust -*-
// error-pattern: Non-exhaustive pattern
enum t { a(u), b }
enum u { c, d }

fn main() {
  let x = a(c);
  alt x {
      a(d) { fail "hello"; }
      b { fail "goodbye"; }
    }
}

