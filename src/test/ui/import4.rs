// error-pattern: import


mod a { pub use b::foo; }
mod b { pub use a::foo; }

fn main() { println!("loop"); }
