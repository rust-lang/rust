// edition:2018
// compile-flags:--extern foo --extern bar

use foo::bar; //~ ERROR unresolved import
use bar::foo;

fn main() {}
