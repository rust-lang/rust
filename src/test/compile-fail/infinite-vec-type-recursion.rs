// -*- rust -*-
// xfail-stage0
// error-pattern: illegal recursive type

type x = vec[x];

fn main() {
  let x b = [];
}
