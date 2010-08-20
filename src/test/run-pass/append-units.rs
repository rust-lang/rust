fn main() {
  auto v = vec(1,2,3);
  v += 4;
  v += 5;
  check (v.(3) == 4);
  check (v.(4) == 5);

  auto s = "hello";
  log s;
  s += 'z' as u8;
  s += 'y' as u8;
  log s;
  check (s.(5) == 'z' as u8);
  check (s.(6) == 'y' as u8);
}
