// xfail-boot
// -*- rust -*-

fn main() {
  auto x = {
    @100
  };

  check (*x == 100);
}
