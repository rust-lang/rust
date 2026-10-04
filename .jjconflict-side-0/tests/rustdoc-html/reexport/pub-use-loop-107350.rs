// This is a regression test for <https://github.com/rust-lang/rust/issues/107350>.
// It shouldn't loop indefinitely.

#![crate_name = "foo"]

//@ has 'foo/oops/enum.OhNo.html'

pub mod oops {
    pub use crate::oops::OhNo;

    mod inner {
        pub enum OhNo {
            Item = 1,
        }
    }

    pub use self::inner::*;
}
