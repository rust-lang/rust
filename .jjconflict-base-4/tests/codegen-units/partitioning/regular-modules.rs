//@ incremental
//@ compile-flags: -Copt-level=0

#![crate_type = "lib"]

// This test ensures that regular fn items and statics are assigned to the CGU of their module.

//~ MONO_ITEM fn foo @@ regular_modules[External]
pub fn foo() {}

//~ MONO_ITEM fn bar @@ regular_modules[External]
pub fn bar() {}

//~ MONO_ITEM static BAZ @@ regular_modules[External]
pub static BAZ: u64 = 0;

pub mod mod1 {

    //~ MONO_ITEM fn mod1::foo @@ regular_modules-mod1[External]
    pub fn foo() {}
    //~ MONO_ITEM fn mod1::bar @@ regular_modules-mod1[External]
    pub fn bar() {}
    //~ MONO_ITEM static mod1::BAZ @@ regular_modules-mod1[External]
    pub static BAZ: u64 = 0;

    pub mod mod1 {
        //~ MONO_ITEM fn mod1::mod1::foo @@ regular_modules-mod1-mod1[External]
        pub fn foo() {}
        //~ MONO_ITEM fn mod1::mod1::bar @@ regular_modules-mod1-mod1[External]
        pub fn bar() {}
        //~ MONO_ITEM static mod1::mod1::BAZ @@ regular_modules-mod1-mod1[External]
        pub static BAZ: u64 = 0;
    }

    pub mod mod2 {
        //~ MONO_ITEM fn mod1::mod2::foo @@ regular_modules-mod1-mod2[External]
        pub fn foo() {}
        //~ MONO_ITEM fn mod1::mod2::bar @@ regular_modules-mod1-mod2[External]
        pub fn bar() {}
        //~ MONO_ITEM static mod1::mod2::BAZ @@ regular_modules-mod1-mod2[External]
        pub static BAZ: u64 = 0;
    }
}

pub mod mod2 {

    //~ MONO_ITEM fn mod2::foo @@ regular_modules-mod2[External]
    pub fn foo() {}
    //~ MONO_ITEM fn mod2::bar @@ regular_modules-mod2[External]
    pub fn bar() {}
    //~ MONO_ITEM static mod2::BAZ @@ regular_modules-mod2[External]
    pub static BAZ: u64 = 0;

    pub mod mod1 {
        //~ MONO_ITEM fn mod2::mod1::foo @@ regular_modules-mod2-mod1[External]
        pub fn foo() {}
        //~ MONO_ITEM fn mod2::mod1::bar @@ regular_modules-mod2-mod1[External]
        pub fn bar() {}
        //~ MONO_ITEM static mod2::mod1::BAZ @@ regular_modules-mod2-mod1[External]
        pub static BAZ: u64 = 0;
    }

    pub mod mod2 {
        //~ MONO_ITEM fn mod2::mod2::foo @@ regular_modules-mod2-mod2[External]
        pub fn foo() {}
        //~ MONO_ITEM fn mod2::mod2::bar @@ regular_modules-mod2-mod2[External]
        pub fn bar() {}
        //~ MONO_ITEM static mod2::mod2::BAZ @@ regular_modules-mod2-mod2[External]
        pub static BAZ: u64 = 0;
    }
}
