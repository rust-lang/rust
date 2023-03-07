//
// We specify incremental here because we want to test the partitioning for
// incremental compilation
// incremental
// compile-flags:-Zprint-mono-items=lazy
// compile-flags:-Zinline-in-all-cgus

#![allow(dead_code)]
#![crate_type="lib"]

mod inline {

    // Important: This function should show up in all codegen units where it is inlined
    //~ MONO_ITEM fn inline::inlined_function @@ local_inlining-user1[Internal] local_inlining-user2[Internal]
    #[inline(always)]
    pub fn inlined_function()
    {

    }
}

pub mod user1 {
    use super::inline;

    //~ MONO_ITEM fn user1::foo @@ local_inlining-user1[External]
    pub fn foo() {
        inline::inlined_function();
    }
}

pub mod user2 {
    use super::inline;

    //~ MONO_ITEM fn user2::bar @@ local_inlining-user2[External]
    pub fn bar() {
        inline::inlined_function();
    }
}

pub mod non_user {

    //~ MONO_ITEM fn non_user::baz @@ local_inlining-non_user[External]
    pub fn baz() {

    }
}
