mod a {
    pub use b::x;
}

mod b {
    pub use a::x; //~ ERROR unresolved import `a::x`

    fn main() { let y = x; }
}

fn main() {}
