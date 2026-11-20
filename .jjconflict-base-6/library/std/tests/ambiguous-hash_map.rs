//! Make sure that a `std` macro `hash_map!` does not cause ambiguity
//! with a local glob import with the same name.
//!
//! See regression https://github.com/rust-lang/rust/issues/147971

mod module {
    macro_rules! hash_map {
        () => {};
    }
    pub(crate) use hash_map;
}

use module::*;

fn main() {
    hash_map! {}
}
