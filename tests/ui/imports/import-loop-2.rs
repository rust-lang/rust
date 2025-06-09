mod a {
    pub use crate::b::x;
}

mod b {
    pub use crate::a::x; //~ ERROR unresolved import `crate::a::x`

    fn main() { let y = x; }
}

fn main() {}
