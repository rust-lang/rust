// error-pattern:Unsatisfied precondition constraint (for example, le(b, d
// xfail-stage0
use std;
import std::str::*;
import std::uint::*;

fn main() {
  let uint a = 1u;
  let uint b = 4u;
  let uint c = 5u;
  // make sure that the constraint le(b, a) exists...
  check le(b, a);
  // ...invalidate it...
  b += 1u;
  check le(c, a);
  // ...and check that it doesn't get set in the poststate of
  // the next statement, since it's not true in the
  // prestate.
  auto d <- a;
  log (safe_slice("kitties", b, d)); 
}