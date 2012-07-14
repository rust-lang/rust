// error-pattern:moop
use std;
import uint;
fn main() { for uint::range(0u, 10u) |_i| { fail ~"moop"; } }
