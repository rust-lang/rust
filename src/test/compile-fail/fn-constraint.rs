// error-pattern: Unsatisfied precondition constraint (for example, le(a, b)
use std;
import std::str::*;

fn main() {
  let uint a = 4u;
  let uint b = 1u;
  log_err (safe_slice("kitties", a, b));
}