// -*- rust -*-

fn main() {
  let str s = "hello";
  s += "world";
  log s;
  check(s.(9) == ('d' as u8));
}
