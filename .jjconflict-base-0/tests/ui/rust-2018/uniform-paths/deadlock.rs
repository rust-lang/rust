//@ edition:2018
//@ compile-flags:--extern foo --extern bar

use bar::foo; //~ ERROR can't find crate for `bar`
use foo::bar;

fn main() {}
