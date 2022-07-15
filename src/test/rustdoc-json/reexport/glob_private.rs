// edition:2018

#![no_core]
#![feature(no_core)]

// @is glob_private.json "$.index[*][?(@.name=='mod1')].kind" \"module\"
// @is glob_private.json "$.index[*][?(@.name=='mod1')].inner.is_stripped" "true"
mod mod1 {
    // @is - "$.index[*][?(@.name=='mod2')].kind" \"module\"
    // @is - "$.index[*][?(@.name=='mod2')].inner.is_stripped" "true"
    mod mod2 {
        // @set m2pub_id = - "$.index[*][?(@.name=='Mod2Public')].id"
        pub struct Mod2Public;

        // @!has - "$.index[*][?(@.name=='Mod2Private')]"
        struct Mod2Private;
    }

    // @has - "$.index[*][?(@.kind=='import' && @.inner.name=='mod2')]"
    pub use self::mod2::*;

    // @set m1pub_id = - "$.index[*][?(@.name=='Mod1Public')].id"
    pub struct Mod1Public;
    // @!has - "$.index[*][?(@.name=='Mod1Private')]"
    struct Mod1Private;
}

// @has - "$.index[*][?(@.kind=='import' && @.inner.name=='mod1')]"
pub use mod1::*;

// @has - "$.index[*][?(@.name=='mod2')].inner.items[*]" $m2pub_id
// @has - "$.index[*][?(@.name=='mod1')].inner.items[*]" $m1pub_id
