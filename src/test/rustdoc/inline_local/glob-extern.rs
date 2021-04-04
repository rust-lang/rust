#![crate_name = "foo"]

mod mod1 {
    extern "C" {
        pub fn public_fn();
        fn private_fn();
    }
}

pub use mod1::*;

// @has foo/index.html
// @!has - "mod1"
// @has - "public_fn"
// @!has - "private_fn"
// @has foo/fn.public_fn.html
// @!has foo/fn.private_fn.html

// @!has foo/mod1/index.html
// @has foo/mod1/fn.public_fn.html
// @!has foo/mod1/fn.private_fn.html
