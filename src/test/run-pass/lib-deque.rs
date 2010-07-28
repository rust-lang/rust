// -*- rust -*-

use std;
import std.deque;

fn main() {
  let deque.t[int] d1 = deque.create[int]();
  check (d1.size() == 0u);
  d1.add_front(17);
  d1.add_front(42);
  d1.add_back(137);
  check (d1.size() == 3u);
  d1.add_back(137);
  check (d1.size() == 4u);
  /* FIXME (issue #133):  We should check that the numbers come back
   * to us correctly once the deque stops zeroing them out. */
}
