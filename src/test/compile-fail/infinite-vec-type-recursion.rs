// xfail-boot
// xfail-stage0
// -*- rust -*-

// error-pattern: infinite recursive type definition

type x = vec[x];

fn main() {
  let x b = vec();
}
