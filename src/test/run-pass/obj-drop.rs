fn main() {
  obj handle(@int i) {
  }
  // This just tests whether the obj leaks its exterior state members.
  auto ob = handle(0xf00f00);
}