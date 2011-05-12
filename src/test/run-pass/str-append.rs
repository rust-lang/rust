// -*- rust -*-

use std;
import std::_str;

fn test1() {
  let str s = "hello";
  s += "world";
  log s;
  assert (s.(9) == ('d' as u8));
}

fn test2() {
  // This tests for issue #163

  let str ff = "abc";
  let str a = ff + "ABC" + ff;
  let str b = "ABC" + ff + "ABC";

  log a;
  log b;

  assert (_str::eq(a, "abcABCabc"));
  assert (_str::eq(b, "ABCabcABC"));
}

fn main() {
  test1();
  test2();
}
