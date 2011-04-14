// xfail-boot
// xfail-stage0
use std;
import std._str;

fn test(str actual, str expected) {
  log actual;
  log expected;
  check (_str.eq(actual, expected));
}

fn main() {
  test(#fmt("hello %d friends and %s things", 10, "formatted"),
    "hello 10 friends and formatted things");

  // Simple tests for types
  test(#fmt("%d", 1), "1");
  test(#fmt("%i", 2), "2");
  test(#fmt("%i", -1), "-1");
  test(#fmt("%u", 10u), "10");
  test(#fmt("%s", "test"), "test");
  test(#fmt("%b", true), "true");
  test(#fmt("%b", false), "false");
  test(#fmt("%c", 'A'), "A");
  test(#fmt("%x", 0xff_u), "ff");
}
