use clippy::a; //~ ERROR unresolved import `clippy`
use clippy::a::b; //~ ERROR failed to resolve: use of unresolved module or unlinked crate `clippy`

use rustdoc::a; //~ ERROR unresolved import `rustdoc`
use rustdoc::a::b; //~ ERROR failed to resolve: use of unresolved module or unlinked crate `rustdoc`

fn main() {}
