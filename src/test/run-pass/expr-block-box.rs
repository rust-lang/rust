// xfail-boot
// xfail-stage0
// -*- rust -*-

fn main() {
  auto x = {
    @100
  };

  check (*x == 100);
}