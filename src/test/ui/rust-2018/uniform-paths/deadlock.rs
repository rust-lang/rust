// edition:2018
// compile-flags:--extern foo --extern bar

use foo::bar; //~ ERROR can't find crate for `foo`
use bar::foo;

fn main() {}
