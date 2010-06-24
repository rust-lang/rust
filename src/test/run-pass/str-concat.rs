// -*- rust -*-

fn main() {
  let str a = "hello";
  let str b = "world";
  let str s = a + b;
  log s;
  check(s.(9) == u8('d'));
}
