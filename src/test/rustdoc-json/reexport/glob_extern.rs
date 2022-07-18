// edition:2018

#![no_core]
#![feature(no_core)]

// @is glob_extern.json "$.index[*][?(@.name=='mod1')].kind" \"module\"
// @is - "$.index[*][?(@.name=='mod1')].inner.is_stripped" "true"
mod mod1 {
    extern "C" {
        // @set public_fn_id = - "$.index[*][?(@.name=='public_fn')].id"
        pub fn public_fn();
        // @!has - "$.index[*][?(@.name=='private_fn')]"
        fn private_fn();
    }
    // @count - "$.index[*][?(@.name=='mod1')].inner.items[*]" 1
    // @has - "$.index[*][?(@.name=='mod1')].inner.items[*]" $public_fn_id
    // @set mod1_id = - "$.index[*][?(@.name=='mod1')].id"
}

// @is - "$.index[*][?(@.kind=='import')].inner.glob" true
// @is - "$.index[*][?(@.kind=='import')].inner.id" $mod1_id
pub use mod1::*;
