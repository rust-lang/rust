// Regression test for #56128. When this `pub(super) use...` gets
// exploded in the HIR, we were not handling IDs correctly.
//
// compile-pass

mod bar {
    pub(super) use self::baz::{x, y};

    mod baz {
        pub fn x() { }
        pub fn y() { }
    }
}

fn main() { }
