
fn main() {
  auto s = "hello";
  let u8 c = s.(4);
  log c;
  check (c == (0x6f as u8));
}
