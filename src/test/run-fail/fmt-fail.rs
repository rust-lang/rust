// xfail-stage0
// error-pattern:meh
use std;
import std::str;

fn main() {
  let str str_var = "meh";
  fail #fmt("%s", str_var);
}