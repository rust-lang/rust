// -*- rust -*-
// error-pattern: non-exhaustive patterns
enum t { a(u), b }
enum u { c, d }

fn main() {
  let x = a(c);
  match x {
      a(d) => { fail ~"hello"; }
      b => { fail ~"goodbye"; }
    }
}

