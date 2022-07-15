// edition:2018

#![no_core]
#![feature(no_core)]

// @is glob_extern.json "$.index[*][?(@.name=='mod1')].kind" \"module\"
// @is glob_extern.json "$.index[*][?(@.name=='mod1')].inner.is_stripped" "true"
mod mod1 {
    extern "C" {
        // @has - "$.index[*][?(@.name=='public_fn')].id"
        pub fn public_fn();
        // @!has - "$.index[*][?(@.name=='private_fn')]"
        fn private_fn();
    }
}

// @is - "$.index[*][?(@.kind=='import')].inner.glob" true
pub use mod1::*;
