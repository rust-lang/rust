// edition:2018

#![no_core]
#![feature(no_core)]

// @!has glob_extern.json "$.index[*][?(@.name=='mod1')]"
mod mod1 {
    extern "C" {
        // @set public_fn_id = - "$.index[*][?(@.name=='public_fn')].id"
        pub fn public_fn();
        // @!has - "$.index[*][?(@.name=='private_fn')]"
        fn private_fn();
    }
}

// @has - "$.index[*][?(@.name=='glob_extern')].inner.items[*]" $public_fn_id
pub use mod1::*;
