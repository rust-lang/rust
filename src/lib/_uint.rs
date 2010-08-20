import std.sys;

fn add(uint x, uint y) -> uint { ret x + y; }
fn sub(uint x, uint y) -> uint { ret x - y; }
fn mul(uint x, uint y) -> uint { ret x * y; }
fn div(uint x, uint y) -> uint { ret x / y; }
fn rem(uint x, uint y) -> uint { ret x % y; }

fn lt(uint x, uint y) -> bool { ret x < y; }
fn le(uint x, uint y) -> bool { ret x <= y; }
fn eq(uint x, uint y) -> bool { ret x == y; }
fn ne(uint x, uint y) -> bool { ret x != y; }
fn ge(uint x, uint y) -> bool { ret x >= y; }
fn gt(uint x, uint y) -> bool { ret x > y; }

iter range(mutable uint lo, uint hi) -> uint {
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

fn to_str(mutable uint n, uint radix) -> str
{
  check (0u < radix && radix <= 16u);
  fn digit(uint n) -> char {
    alt (n) {
      case (0u) { ret '0'; }
      case (1u) { ret '1'; }
      case (2u) { ret '2'; }
      case (3u) { ret '3'; }
      case (4u) { ret '4'; }
      case (5u) { ret '5'; }
      case (6u) { ret '6'; }
      case (7u) { ret '7'; }
      case (8u) { ret '8'; }
      case (9u) { ret '9'; }
      case (10u) { ret 'a'; }
      case (11u) { ret 'b'; }
      case (12u) { ret 'c'; }
      case (13u) { ret 'd'; }
      case (14u) { ret 'e'; }
      case (15u) { ret 'f'; }
    }
  }

  if (n == 0u) { ret "0"; }

  let uint r = 1u;
  if (n > r) {
    while ((r*radix) < n) {
      r *= radix;
    }
  }

  let str s = "";
  while (n > 0u) {

    auto i = n/r;

    n -= (i * r);
    r /= radix;

    s += digit(i) as u8;
  }

  while (r > 0u) {
    s += '0' as u8;
    r /= radix;
  }

  ret s;
}
