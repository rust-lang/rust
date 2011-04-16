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
  test(#fmt("%X", 0x12ab_u), "12AB");
  test(#fmt("%t", 0b11010101_u), "11010101");

  // 32-bit limits
  test(#fmt("%i", -2147483648), "-2147483648");
  test(#fmt("%i", 2147483647), "2147483647");
  test(#fmt("%u", 4294967295u), "4294967295");
  test(#fmt("%x", 0xffffffff_u), "ffffffff");
  test(#fmt("%t", 0xffffffff_u), "11111111111111111111111111111111");

  // Widths
  test(#fmt("%10d", 500), "       500");
  test(#fmt("%10d", -500), "      -500");
  test(#fmt("%10u", 500u), "       500");
  test(#fmt("%10s", "test"), "      test");
  test(#fmt("%10b", true), "      true");
  test(#fmt("%10x", 0xff_u), "        ff");
  test(#fmt("%10X", 0xff_u), "        FF");
  test(#fmt("%10t", 0xff_u), "  11111111");
}
