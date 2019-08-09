// We specify -Z incremental here because we want to test the partitioning for
// incremental compilation
// compile-flags:-Zprint-mono-items=lazy -Zincremental=tmp/partitioning-tests/statics

#![crate_type="rlib"]

//~ MONO_ITEM static statics::FOO[0] @@ statics[Internal]
static FOO: u32 = 0;

//~ MONO_ITEM static statics::BAR[0] @@ statics[Internal]
static BAR: u32 = 0;

//~ MONO_ITEM fn statics::function[0] @@ statics[External]
pub fn function() {
    //~ MONO_ITEM static statics::function[0]::FOO[0] @@ statics[Internal]
    static FOO: u32 = 0;

    //~ MONO_ITEM static statics::function[0]::BAR[0] @@ statics[Internal]
    static BAR: u32 = 0;
}

pub mod mod1 {
    //~ MONO_ITEM static statics::mod1[0]::FOO[0] @@ statics-mod1[Internal]
    static FOO: u32 = 0;

    //~ MONO_ITEM static statics::mod1[0]::BAR[0] @@ statics-mod1[Internal]
    static BAR: u32 = 0;

    //~ MONO_ITEM fn statics::mod1[0]::function[0] @@ statics-mod1[External]
    pub fn function() {
        //~ MONO_ITEM static statics::mod1[0]::function[0]::FOO[0] @@ statics-mod1[Internal]
        static FOO: u32 = 0;

        //~ MONO_ITEM static statics::mod1[0]::function[0]::BAR[0] @@ statics-mod1[Internal]
        static BAR: u32 = 0;
    }
}
