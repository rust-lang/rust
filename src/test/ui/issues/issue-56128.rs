// Regression test for #56128. When this `pub(super) use...` gets
// exploded in the HIR, we were not handling ids correctly.
//
// build-pass (FIXME(62277): could be check-pass?)

mod bar {
    pub(super) use self::baz::{x, y};

    mod baz {
        pub fn x() { }
        pub fn y() { }
    }
}

fn main() { }
