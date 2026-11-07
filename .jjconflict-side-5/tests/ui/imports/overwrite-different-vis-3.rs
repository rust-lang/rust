// Regression test for issue #152606.

//@ check-pass

mod outer {
    mod inner {
        use super::*; // should go before the ambiguous glob imports
    }

    use crate::*;
    pub use crate::*;
}

fn main() {}
