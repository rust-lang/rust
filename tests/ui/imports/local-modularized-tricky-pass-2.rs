//@ check-pass
//
// `#[macro_export] macro_rules` that doesn't originate from macro expansions can be placed
// into the root module soon enough to act as usual items and shadow globs and preludes.

#![feature(decl_macro)]
#![allow(non_local_definitions)]

// `macro_export` shadows globs
use inner1::*;

mod inner1 {
    pub macro exported() {}
}

exported!();

mod deep {
    fn deep() {
        type Deeper = [u8; {
            #[macro_export]
            macro_rules! exported {
                () => ( struct Б; )
            }

            0
        }];
    }
}

// `macro_export` shadows std prelude
fn main() {
    panic!();
}

mod inner3 {
    #[macro_export]
    macro_rules! panic {
        () => ( struct Г; )
    }
}

// `macro_export` shadows builtin macros
include!();

mod inner4 {
    #[macro_export]
    macro_rules! include {
        () => ( struct Д; )
    }
}
