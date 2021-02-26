// edition:2018

#![no_core]
#![feature(no_core)]
mod mod1 {
    extern "C" {
        pub fn public_fn();
        fn private_fn();
    }
}

pub use mod1::*;

// @!has glob_extern.json "$.index[*][?(@.name=='mod1')]"
// @!has - "$.index[*][?(@.name=='private_fn')]"
// @set public_fn_id = - "$.index[*][?(@.name=='public_fn')].id"
// @has - "$.index[*][?(@.name=='glob_extern')].inner.items[*]" $public_fn_id
