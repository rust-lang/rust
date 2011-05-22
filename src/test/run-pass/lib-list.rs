use std;
import std::list;
import std::list::car;
import std::list::cdr;
import std::list::from_vec;

fn test_from_vec() {
  auto l = from_vec([0, 1, 2]);
  assert (car(l) == 0);
  assert (car(cdr(l)) == 1);
  assert (car(cdr(cdr(l))) == 2);
}

fn main() {
  test_from_vec();
}
