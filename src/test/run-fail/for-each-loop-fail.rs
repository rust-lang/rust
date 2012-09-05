// error-pattern:moop
use std;
fn main() { for uint::range(0u, 10u) |_i| { fail ~"moop"; } }
