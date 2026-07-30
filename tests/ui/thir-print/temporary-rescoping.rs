//! Test that temporary rescoping operators don't appear in THIR, apart from modifying temporary
//! scopes within their operands.
//@ compile-flags: -Zunpretty=thir-tree
//@ check-pass

#![feature(temporary_rescoping)]

use std::temporary_rescoping::{scope, extend};

fn main() {
    let _ = 'a: {
        &scope!('a => ());
        &extend!('a => ());
    };
}
