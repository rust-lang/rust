// error-pattern:moop
use std;
import std::uint;
fn main() { uint::range(0u, 10u) {|_i| fail "moop"; } }
