// Check that cyclic glob imports are allowed with underscore imports

// check-pass

mod x {
    pub use crate::y::*;
    pub use std::ops::Deref as _;
}

mod y {
    pub use crate::x::*;
    pub use std::ops::Deref as _;
}

pub fn main() {
    use x::*;
    #[allow(noop_method_call)]
    (&0).deref();
}
