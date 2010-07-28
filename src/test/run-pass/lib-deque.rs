// -*- rust -*-

use std;
import std.deque;

fn test_simple() {
  let deque.t[int] d = deque.create[int]();
  check (d.size() == 0u);
  d.add_front(17);
  d.add_front(42);
  d.add_back(137);
  check (d.size() == 3u);
  d.add_back(137);
  check (d.size() == 4u);

  log d.peek_front();
  check (d.peek_front() == 42);

  log d.peek_back();
  check (d.peek_back() == 137);

  let int i = d.pop_front();
  log i;
  check (i == 42);

  i = d.pop_back();
  log i;
  check (i == 137);

  i = d.pop_back();
  log i;
  check (i == 137);

  i = d.pop_back();
  log i;
  check (i == 17);

  /* FIXME (issue #138):  Test d.get() once it no longer causes
   * segfault. */
}

fn main() {
  test_simple();
}
