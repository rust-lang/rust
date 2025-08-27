mod a { pub use crate::b::foo; } //~ ERROR unresolved import `crate::b`

fn main() { println!("loop"); }
