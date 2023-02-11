// We specify incremental here because we want to test the partitioning for
// incremental compilation
// incremental
// compile-flags:-Zprint-mono-items=eager

#![allow(dead_code)]
#![crate_type="lib"]

//~ MONO_ITEM fn foo @@ regular_modules[Internal]
fn foo() {}

//~ MONO_ITEM fn bar @@ regular_modules[Internal]
fn bar() {}

//~ MONO_ITEM static BAZ @@ regular_modules[Internal]
static BAZ: u64 = 0;

mod mod1 {

    //~ MONO_ITEM fn mod1::foo @@ regular_modules-mod1[Internal]
    fn foo() {}
    //~ MONO_ITEM fn mod1::bar @@ regular_modules-mod1[Internal]
    fn bar() {}
    //~ MONO_ITEM static mod1::BAZ @@ regular_modules-mod1[Internal]
    static BAZ: u64 = 0;

    mod mod1 {
        //~ MONO_ITEM fn mod1::mod1::foo @@ regular_modules-mod1-mod1[Internal]
        fn foo() {}
        //~ MONO_ITEM fn mod1::mod1::bar @@ regular_modules-mod1-mod1[Internal]
        fn bar() {}
        //~ MONO_ITEM static mod1::mod1::BAZ @@ regular_modules-mod1-mod1[Internal]
        static BAZ: u64 = 0;
    }

    mod mod2 {
        //~ MONO_ITEM fn mod1::mod2::foo @@ regular_modules-mod1-mod2[Internal]
        fn foo() {}
        //~ MONO_ITEM fn mod1::mod2::bar @@ regular_modules-mod1-mod2[Internal]
        fn bar() {}
        //~ MONO_ITEM static mod1::mod2::BAZ @@ regular_modules-mod1-mod2[Internal]
        static BAZ: u64 = 0;
    }
}

mod mod2 {

    //~ MONO_ITEM fn mod2::foo @@ regular_modules-mod2[Internal]
    fn foo() {}
    //~ MONO_ITEM fn mod2::bar @@ regular_modules-mod2[Internal]
    fn bar() {}
    //~ MONO_ITEM static mod2::BAZ @@ regular_modules-mod2[Internal]
    static BAZ: u64 = 0;

    mod mod1 {
        //~ MONO_ITEM fn mod2::mod1::foo @@ regular_modules-mod2-mod1[Internal]
        fn foo() {}
        //~ MONO_ITEM fn mod2::mod1::bar @@ regular_modules-mod2-mod1[Internal]
        fn bar() {}
        //~ MONO_ITEM static mod2::mod1::BAZ @@ regular_modules-mod2-mod1[Internal]
        static BAZ: u64 = 0;
    }

    mod mod2 {
        //~ MONO_ITEM fn mod2::mod2::foo @@ regular_modules-mod2-mod2[Internal]
        fn foo() {}
        //~ MONO_ITEM fn mod2::mod2::bar @@ regular_modules-mod2-mod2[Internal]
        fn bar() {}
        //~ MONO_ITEM static mod2::mod2::BAZ @@ regular_modules-mod2-mod2[Internal]
        static BAZ: u64 = 0;
    }
}
