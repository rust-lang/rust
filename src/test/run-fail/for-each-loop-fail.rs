// xfail-stage0
// error-pattern:moop
use std;
import std::uint;
fn main() { for each i: uint  in uint::range(0u, 10u) { fail "moop"; } }