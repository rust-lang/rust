// We specify incremental here because we want to test the partitioning for
// incremental compilation
// incremental
// compile-flags:-Zprint-mono-items=lazy

#![crate_type="rlib"]

//~ MONO_ITEM static FOO @@ statics[Internal]
static FOO: u32 = 0;

//~ MONO_ITEM static BAR @@ statics[Internal]
static BAR: u32 = 0;

//~ MONO_ITEM fn function @@ statics[External]
pub fn function() {
    //~ MONO_ITEM static function::FOO @@ statics[Internal]
    static FOO: u32 = 0;

    //~ MONO_ITEM static function::BAR @@ statics[Internal]
    static BAR: u32 = 0;
}

pub mod mod1 {
    //~ MONO_ITEM static mod1::FOO @@ statics-mod1[Internal]
    static FOO: u32 = 0;

    //~ MONO_ITEM static mod1::BAR @@ statics-mod1[Internal]
    static BAR: u32 = 0;

    //~ MONO_ITEM fn mod1::function @@ statics-mod1[External]
    pub fn function() {
        //~ MONO_ITEM static mod1::function::FOO @@ statics-mod1[Internal]
        static FOO: u32 = 0;

        //~ MONO_ITEM static mod1::function::BAR @@ statics-mod1[Internal]
        static BAR: u32 = 0;
    }
}
