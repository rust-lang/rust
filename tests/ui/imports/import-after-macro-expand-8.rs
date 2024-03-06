//@ check-pass
// https://github.com/rust-lang/rust/pull/113242#issuecomment-1616034904

mod a {
    pub trait P {}
}
pub use a::*;

mod b {
    #[derive(Clone)]
    pub enum P {
        A
    }
}
pub use b::P;

mod c {
    use crate::*;
    pub struct S(Vec<P>);
}

fn main() {}
