//@ edition:2018

//@ is "$.index[?(@.name=='mod1')].inner.module.is_stripped" "true"
mod mod1 {
    //@ is "$.index[?(@.name=='mod2')].inner.module.is_stripped" "true"
    mod mod2 {
        //@ set m2pub_id = "$.index[?(@.name=='Mod2Public')].id"
        pub struct Mod2Public;

        //@ !has "$.index[?(@.name=='Mod2Private')]"
        struct Mod2Private;
    }

    //@ set mod2_use_id = "$.index[?(@.docs=='Mod2 re-export')].id"
    //@ is "$.index[?(@.docs=='Mod2 re-export')].inner.use.name" \"mod2\"
    /// Mod2 re-export
    pub use self::mod2::*;

    //@ set m1pub_id = "$.index[?(@.name=='Mod1Public')].id"
    pub struct Mod1Public;
    //@ !has "$.index[?(@.name=='Mod1Private')]"
    struct Mod1Private;
}

//@ set mod1_use_id = "$.index[?(@.docs=='Mod1 re-export')].id"
//@ is "$.index[?(@.docs=='Mod1 re-export')].inner.use.name" \"mod1\"
/// Mod1 re-export
pub use mod1::*;

//@ ismany "$.index[?(@.name=='mod2')].inner.module.items[*]" $m2pub_id
//@ ismany "$.index[?(@.name=='mod1')].inner.module.items[*]" $m1pub_id $mod2_use_id
//@ ismany "$.index[?(@.name=='glob_private')].inner.module.items[*]" $mod1_use_id
