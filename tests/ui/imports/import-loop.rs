use y::x;

mod y {
    pub use crate::y::x; //~ ERROR unresolved import `crate::y::x`
}

fn main() { }
