mod a {
    pub use crate::b::x; //~ ERROR unresolved import `crate::b::x`
}

mod b {
    pub use crate::a::x;

    fn main() { let y = x; }
}

fn main() {}
