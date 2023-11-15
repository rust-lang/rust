// edition:2018

// @is "$.index[*][?(@.name=='mod1')].inner.module.is_stripped" "true"
mod mod1 {
    extern "C" {
        // @set public_fn_id = "$.index[*][?(@.name=='public_fn')].id"
        pub fn public_fn();
        // @!has "$.index[*][?(@.name=='private_fn')]"
        fn private_fn();
    }
    // @ismany "$.index[*][?(@.name=='mod1')].inner.module.items[*]" $public_fn_id
    // @set mod1_id = "$.index[*][?(@.name=='mod1')].id"
}

// @is "$.index[*][?(@.inner.import)].inner.import.glob" true
// @is "$.index[*][?(@.inner.import)].inner.import.id" $mod1_id
// @set use_id = "$.index[*][?(@.inner.import)].id"
// @ismany "$.index[*][?(@.name=='glob_extern')].inner.module.items[*]" $use_id
pub use mod1::*;
