// https://github.com/rust-lang/rust/pull/113242#issuecomment-1616034904
// similar with `import-after-macro-expand-2.rs`

mod a {
    pub trait P {}
}

pub use a::*;

mod c {
    use crate::*;
    pub struct S(Vec<P>);
    //~^ ERROR `P` is ambiguous
    //~| ERROR the size for values of type `(dyn a::P + 'static)` cannot be known at compilation time
    //~| WARN trait objects without an explicit `dyn` are deprecated
    //~| WARN trait objects without an explicit `dyn` are deprecated
    //~| WARN trait objects without an explicit `dyn` are deprecated
    //~| WARN this is accepted in the current edition
    //~| WARN this is accepted in the current edition
    //~| WARN this is accepted in the current edition
}

#[derive(Clone)]
pub enum P {
    A
}

fn main() {}
