// xfail-boot
// xfail-stage0
// xfail-stage1
// xfail-stage2
// -*- rust -*-

// error-pattern: infinite recursive type definition

type x = vec[x];

fn main() {
  let x b = vec();
}
