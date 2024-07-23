use clippy::a; //~ ERROR unresolved import `clippy`
use clippy::a::b; //~ ERROR cannot find item `clippy`

use rustdoc::a; //~ ERROR unresolved import `rustdoc`
use rustdoc::a::b; //~ ERROR cannot find item `rustdoc`

fn main() {}
