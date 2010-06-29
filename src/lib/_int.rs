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

iter range(mutable int lo, int hi) -> int {
  while (lo < hi) {
    put lo;
    lo += 1;
  }
}

iter urange(mutable uint lo, uint hi) -> uint {
  while (lo < hi) {
    put lo;
    lo += uint(1);
  }
}
