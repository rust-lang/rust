//@ check-pass
// Tests that we properly handle a nested macro expansion
// involving a `#[doc]` attribute
#![deny(missing_docs)]
//! Crate docs

macro_rules! doc_comment {
    ($x:expr, $($tt:tt)*) => {
        #[doc = $x]
        $($tt)*
    }
}

macro_rules! make_comment {
    () => {
        doc_comment!("Function docs",
            pub fn bar() {}
        );
    }
}


make_comment!();

fn main() {}
