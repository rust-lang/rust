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
  test(#fmt("d: %d", 1), "d: 1");
  test(#fmt("i: %i", 2), "i: 2");
  test(#fmt("s: %s", "test"), "s: test");
}
