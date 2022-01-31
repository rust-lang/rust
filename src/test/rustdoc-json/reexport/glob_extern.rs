// edition:2018

#![no_core]
#![feature(no_core)]

// @is glob_extern.json "$.index[*][?(@.name=='mod1')].kind" \"stripped_module\"
mod mod1 {
    extern "C" {
        // @has - "$.index[*][?(@.name=='public_fn')]"
        pub fn public_fn();
        // @!has - "$.index[*][?(@.name=='private_fn')]"
        fn private_fn();
    }
}

// @is - "$.index[*][?(@.kind=='import')].inner.glob" true
pub use mod1::*;
