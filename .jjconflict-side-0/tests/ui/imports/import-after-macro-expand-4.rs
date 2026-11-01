//@ check-pass
// https://github.com/rust-lang/rust/pull/113242#issuecomment-1616034904
// similar with `import-after-macro-expand-2.rs`

mod a {
    pub trait P {}
}

pub use a::*;

mod c {
    use crate::*;
    pub struct S(Vec<P>);
}

#[derive(Clone)]
pub enum P {
    A
}

fn main() {}
