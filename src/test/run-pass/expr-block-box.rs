// xfail-boot
// -*- rust -*-

fn main() {
  auto x = {
    @100
  };

  assert (*x == 100);
}
