// edition:2018

#![no_core]
#![feature(no_core)]

mod mod1 {
    mod mod2 {
        pub struct Mod2Public;
        struct Mod2Private;
    }
    pub use self::mod2::*;

    pub struct Mod1Public;
    struct Mod1Private;
}
pub use mod1::*;

// @set m2pub_id = glob_private.json "$.index[*][?(@.name=='Mod2Public')].id"
// @set m1pub_id = - "$.index[*][?(@.name=='Mod1Public')].id"
// @has - "$.index[*][?(@.name=='glob_private')].inner.items[*]" $m2pub_id
// @has - "$.index[*][?(@.name=='glob_private')].inner.items[*]" $m1pub_id
// @!has - "$.index[*][?(@.name=='mod1')]"
// @!has - "$.index[*][?(@.name=='mod2')]"
// @!has - "$.index[*][?(@.name=='Mod1Private')]"
// @!has - "$.index[*][?(@.name=='Mod2Private')]"
