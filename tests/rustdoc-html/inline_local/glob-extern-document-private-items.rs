//@ compile-flags: --document-private-items

#![crate_name = "foo"]

mod mod1 {
    extern "C" {
        pub fn public_fn();
        fn private_fn();
    }
}

pub use mod1::*;

//@ has foo/index.html
//@ hasraw - "mod1"
//@ hasraw - "public_fn"
//@ !hasraw - "private_fn"
//@ has foo/fn.public_fn.html
//@ !has foo/fn.private_fn.html

//@ has foo/mod1/index.html
//@ hasraw - "public_fn"
//@ hasraw - "private_fn"
//@ has foo/mod1/fn.public_fn.html
//@ has foo/mod1/fn.private_fn.html
