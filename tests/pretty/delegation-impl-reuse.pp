#![feature(prelude_import)]
#![no_std]
//@ pretty-compare-only
//@ pretty-mode:expanded
//@ pp-exact:delegation-impl-reuse.pp

#![allow(incomplete_features)]
#![feature(fn_delegation)]
extern crate std;
#[prelude_import]
use ::std::prelude::rust_2015::*;

trait Trait<T> {
    fn foo(&self) {}
    fn bar(&self) {}
    fn baz(&self) {}
}

struct S;

impl Trait<{
        struct S;
        0
    }> for S {
    reuse Trait<{
            struct S;
            0
        }>::foo {
        self.0
    }
    reuse Trait<{
            struct S;
            0
        }>::bar {
        self.0
    }
    reuse Trait<{
            struct S;
            0
        }>::baz {
        self.0
    }
}

fn main() {}
