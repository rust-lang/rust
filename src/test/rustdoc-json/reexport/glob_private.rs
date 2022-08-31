// edition:2018

#![no_core]
#![feature(no_core)]

// @is "$.index[*][?(@.name=='mod1')].kind" \"module\"
// @is "$.index[*][?(@.name=='mod1')].inner.is_stripped" "true"
mod mod1 {
    // @is "$.index[*][?(@.name=='mod2')].kind" \"module\"
    // @is "$.index[*][?(@.name=='mod2')].inner.is_stripped" "true"
    mod mod2 {
        // @set m2pub_id = "$.index[*][?(@.name=='Mod2Public')].id"
        pub struct Mod2Public;

        // @!has "$.index[*][?(@.name=='Mod2Private')]"
        struct Mod2Private;
    }

    // @set mod2_use_id = "$.index[*][?(@.kind=='import' && @.inner.name=='mod2')].id"
    pub use self::mod2::*;

    // @set m1pub_id = "$.index[*][?(@.name=='Mod1Public')].id"
    pub struct Mod1Public;
    // @!has "$.index[*][?(@.name=='Mod1Private')]"
    struct Mod1Private;
}

// @set mod1_use_id = "$.index[*][?(@.kind=='import' && @.inner.name=='mod1')].id"
pub use mod1::*;

// @ismany "$.index[*][?(@.name=='mod2')].inner.items[*]" $m2pub_id
// @ismany "$.index[*][?(@.name=='mod1')].inner.items[*]" $m1pub_id $mod2_use_id
// @ismany "$.index[*][?(@.name=='glob_private')].inner.items[*]" $mod1_use_id
