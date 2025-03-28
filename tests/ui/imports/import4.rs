mod a { pub use b::foo; }
mod b { pub use a::foo; } //~ ERROR unresolved import `a::foo`

fn main() { println!("loop"); }
