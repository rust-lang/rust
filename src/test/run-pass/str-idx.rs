
fn main() {
  auto s = "hello";
  let u8 c = s.(4);
  log c;
  check (c == u8(0x6f));
}
