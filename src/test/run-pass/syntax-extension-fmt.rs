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
}
