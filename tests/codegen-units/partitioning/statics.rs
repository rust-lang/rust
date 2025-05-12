//@ incremental
//@ compile-flags: -Copt-level=0

#![crate_type = "rlib"]

// This test ensures that statics are assigned to the correct module when they are defined inside
// of a function.

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
