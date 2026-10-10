//@ incremental
//@ compile-flags: -Copt-level=0

#![crate_type = "lib"]

// This test checks that a monomorphic inline(always) function is instantiated in every CGU that
// references it, even though this is an unoptimized incremental build.
// It also checks that an inline(always) function is only placed in CGUs that reference it.

mod inline {
    //~ MONO_ITEM fn inline::inlined_function @@ inline_always-user1[Internal] inline_always-user2[Internal]
    #[inline(always)]
    pub fn inlined_function() {}
}

pub mod user1 {
    use super::inline;

    //~ MONO_ITEM fn user1::foo @@ inline_always-user1[External]
    pub fn foo() {
        inline::inlined_function();
    }
}

pub mod user2 {
    use super::inline;

    //~ MONO_ITEM fn user2::bar @@ inline_always-user2[External]
    pub fn bar() {
        inline::inlined_function();
    }
}

pub mod non_user {

    //~ MONO_ITEM fn non_user::baz @@ inline_always-non_user[External]
    pub fn baz() {}
}
