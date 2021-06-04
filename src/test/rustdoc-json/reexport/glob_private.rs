// edition:2018

#![no_core]
#![feature(no_core)]

// @!has glob_private.json "$.index[*][?(@.name=='mod1')]"
mod mod1 {
    // @!has - "$.index[*][?(@.name=='mod2')]"
    mod mod2 {
        // @set m2pub_id = - "$.index[*][?(@.name=='Mod2Public')].id"
        pub struct Mod2Public;

        // @!has - "$.index[*][?(@.name=='Mod2Private')]"
        struct Mod2Private;
    }
    pub use self::mod2::*;

    // @set m1pub_id = - "$.index[*][?(@.name=='Mod1Public')].id"
    pub struct Mod1Public;

    // @!has - "$.index[*][?(@.name=='Mod1Private')]"
    struct Mod1Private;
}
pub use mod1::*;

// @has - "$.index[*][?(@.name=='glob_private')].inner.items[*]" $m2pub_id
// @has - "$.index[*][?(@.name=='glob_private')].inner.items[*]" $m1pub_id
