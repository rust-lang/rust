// check-pass
// aux-build: issue-114682-3-extern.rs

// https://github.com/rust-lang/rust/pull/114682#issuecomment-1880625909
extern crate issue_114682_3_extern;

use issue_114682_3_extern::*;

mod auto {
    pub trait SettingsExt {
        fn ext(&self) {}
    }

    impl<T> SettingsExt for T {}
}

pub use self::auto::*;
//~^ WARNING ambiguous glob re-exports

pub fn main() {
    let a: u8 = 1;
    a.ext();
}
