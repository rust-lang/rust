// check-only
// run-rustfix
// edition:2018

// https://github.com/rust-lang/rust/issues/73948
// Tests that when `meta` is used as a module name and imported uniformly a
// suggestion is made to make the import unambiguous with `crate::meta`

mod meta {
    pub const FOO: bool = true;
}

use meta::FOO; //~ ERROR can't find crate for `meta`

fn main() {
    assert!(FOO);
}
