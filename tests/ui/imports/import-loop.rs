use y::x;

mod y {
    pub use y::x; //~ ERROR unresolved import `y::x`
}

fn main() { }
