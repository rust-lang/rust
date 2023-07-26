// https://github.com/rust-lang/rust/pull/113242#issuecomment-1616034904
// similar with `import-after-macro-expand-2.rs`

mod a {
    pub trait P {}
}

pub use a::*;

mod c {
    use crate::*;
    pub struct S(Vec<P>);
    //~^ ERROR the size for values of type
    //~| WARNING trait objects without an explicit
    //~| WARNING this is accepted in the current edition
    //~| WARNING trait objects without an explicit
    //~| WARNING this is accepted in the current edition
    //~| WARNING trait objects without an explicit
    //~| WARNING this is accepted in the current edition

    // FIXME: should works, but doesn't currently refer
    // to it due to backward compatibility
}

#[derive(Clone)]
pub enum P {
    A
}

fn main() {}
