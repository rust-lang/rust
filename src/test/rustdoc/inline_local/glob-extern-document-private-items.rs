// compile-flags: --document-private-items

#![crate_name = "foo"]

mod mod1 {
    extern "C" {
        pub fn public_fn();
        fn private_fn();
    }
}

pub use mod1::*;

// @has foo/index.html
// @hastext - "mod1"
// @hastext - "public_fn"
// @!has - "private_fn"
// @has foo/fn.public_fn.html
// @!has foo/fn.private_fn.html

// @has foo/mod1/index.html
// @hastext - "public_fn"
// @hastext - "private_fn"
// @has foo/mod1/fn.public_fn.html
// @has foo/mod1/fn.private_fn.html
