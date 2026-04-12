//@ check-pass

mod m {
    pub struct S {}
}

mod one_private {
    use crate::m::*;
    pub use crate::m::*;
}

// One of the ambiguous imports is not visible from here,
// but it still contributes to the ambiguity.
use crate::one_private::S;

// Separate module to make visibilities `in crate::inner` and `in crate::one_private` unordered.
mod inner {
    // One of the ambiguous imports is not visible from here,
    // but it still contributes to the ambiguity.
    use crate::one_private::S;
}

fn main() {}
