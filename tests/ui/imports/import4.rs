mod a { pub use crate::b::foo; }
mod b { pub use crate::a::foo; } //~ ERROR unresolved import `crate::a::foo`

fn main() { println!("loop"); }
