// Anti-regression test to demonstrate that at least we mitigated breakage from adding a new
// `#[align]` built-in attribute.
//
// See https://github.com/rust-lang/rust/issues/143834.

//@ check-pass

// Needs edition >= 2018 macro use behavior.
//@ edition: 2018

macro_rules! align {
    () => {
        /* .. */
    };
}

pub(crate) use align;

fn main() {}
