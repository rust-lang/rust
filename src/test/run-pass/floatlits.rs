// xfail-boot
fn main() {
  auto f = 4.999999999999;
  check (f > 4.90);
  check (f < 5.0);
  auto g = 4.90000000001e-10;
  check(g > 5e-11);
  check(g < 5e-9);
}