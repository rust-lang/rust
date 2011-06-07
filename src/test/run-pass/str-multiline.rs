// -*- rust -*-

use std;
import std::str;

fn main() {
  let str a = "this \
is a test";
  let str b = "this \
               is \
               another \
               test";
  assert (str::eq(a, "this is a test"));
  assert (str::eq(b, "this is another test"));
}