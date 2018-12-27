// ignore-tidy-linelength
// We specify -Z incremental here because we want to test the partitioning for
// incremental compilation
// compile-flags:-Zprint-mono-items=lazy -Zincremental=tmp/partitioning-tests/local-inlining
// compile-flags:-Zinline-in-all-cgus

#![allow(dead_code)]
#![crate_type="lib"]

mod inline {

    // Important: This function should show up in all codegen units where it is inlined
    //~ MONO_ITEM fn local_inlining::inline[0]::inlined_function[0] @@ local_inlining-user1[Internal] local_inlining-user2[Internal]
    #[inline(always)]
    pub fn inlined_function()
    {

    }
}

pub mod user1 {
    use super::inline;

    //~ MONO_ITEM fn local_inlining::user1[0]::foo[0] @@ local_inlining-user1[External]
    pub fn foo() {
        inline::inlined_function();
    }
}

pub mod user2 {
    use super::inline;

    //~ MONO_ITEM fn local_inlining::user2[0]::bar[0] @@ local_inlining-user2[External]
    pub fn bar() {
        inline::inlined_function();
    }
}

pub mod non_user {

    //~ MONO_ITEM fn local_inlining::non_user[0]::baz[0] @@ local_inlining-non_user[External]
    pub fn baz() {

    }
}
