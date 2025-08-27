mod a { pub use crate::b::foo; } //~ ERROR unresolved import `crate::b::foo`
mod b { pub use crate::a::foo; }

fn main() { println!("loop"); }
