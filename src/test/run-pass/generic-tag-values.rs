// -*- rust -*-

type noption[T] = tag(some(T));

fn main() {
  let noption[int] nop = some[int](5);
  alt (nop) {
    case (some[int](n)) {
      log n;
      check (n == 5);
    }
  }

  let noption[tup(int, int)] nop2 = some[tup(int, int)](tup(17, 42));
  alt (nop2) {
    case (some[tup(int, int)](t)) {
      log t._0;
      log t._1;
      check (t._0 == 17);
      check (t._1 == 42);
    }
  }
}
