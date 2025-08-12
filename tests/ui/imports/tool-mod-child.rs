use clippy::a; //~ ERROR unresolved import `clippy`
use clippy::a::b; //~ ERROR cannot find

use rustdoc::a; //~ ERROR unresolved import `rustdoc`
use rustdoc::a::b; //~ ERROR cannot find

fn main() {}
