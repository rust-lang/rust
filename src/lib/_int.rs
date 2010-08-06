import std.sys;

fn add(int x, int y) -> int { ret x + y; }
fn sub(int x, int y) -> int { ret x - y; }
fn mul(int x, int y) -> int { ret x * y; }
fn div(int x, int y) -> int { ret x / y; }
fn rem(int x, int y) -> int { ret x % y; }

fn lt(int x, int y) -> bool { ret x < y; }
fn le(int x, int y) -> bool { ret x <= y; }
fn eq(int x, int y) -> bool { ret x == y; }
fn ne(int x, int y) -> bool { ret x != y; }
fn ge(int x, int y) -> bool { ret x >= y; }
fn gt(int x, int y) -> bool { ret x > y; }

fn positive(int x) -> bool { ret x > 0; }
fn negative(int x) -> bool { ret x < 0; }
fn nonpositive(int x) -> bool { ret x <= 0; }
fn nonnegative(int x) -> bool { ret x >= 0; }

iter range(mutable int lo, int hi) -> int {
  while (lo < hi) {
    put lo;
    lo += 1;
  }
}

iter urange(mutable uint lo, uint hi) -> uint {
  while (lo < hi) {
    put lo;
    lo += 1u;
  }
}

fn next_power_of_two(uint n) -> uint {
  // FIXME change |* uint(4)| below to |* uint(8) / uint(2)| and watch the
  // world explode.
  let uint halfbits = sys.rustrt.size_of[uint]() * 4u;
  let uint tmp = n - 1u;
  let uint shift = 1u;
  while (shift <= halfbits) {
    tmp |= tmp >> shift;
    shift <<= 1u;
  }
  ret tmp + 1u;
}

fn uto_string(mutable uint n, uint radix) -> str
{
  check (0u < radix && radix <= 16u);
  fn digit(uint n) -> str {
    alt (n) {
      case (0u) { ret "0"; }
      case (1u) { ret "1"; }
      case (2u) { ret "2"; }
      case (3u) { ret "3"; }
      case (4u) { ret "4"; }
      case (5u) { ret "5"; }
      case (6u) { ret "6"; }
      case (7u) { ret "7"; }
      case (8u) { ret "8"; }
      case (9u) { ret "9"; }
      case (10u) { ret "A"; }
      case (11u) { ret "B"; }
      case (12u) { ret "C"; }
      case (13u) { ret "D"; }
      case (14u) { ret "E"; }
      case (15u) { ret "F"; }
    }
  }

  if (n == 0u) { ret "0"; }

  let str s = "";
  while (n > 0u) {
    s = digit(n % radix) + s;
    n /= radix;
  }
  ret s;
}

fn to_string(mutable int n, uint radix) -> str
{
  check (0u < radix && radix <= 16u);
  if (n < 0) {
    ret "-" + uto_string((-n) as uint, radix);
  } else {
    ret uto_string(n as uint, radix);
  }
}
