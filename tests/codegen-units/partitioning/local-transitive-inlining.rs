//@ incremental
//@ compile-flags: -Copt-level=0

#![crate_type = "rlib"]

// This test checks that a monomorphic inline(always) function is instantiated in every CGU that
// references it, even if it is only referenced via another module.
// The modules `inline` and `direct_user` do not get CGUs because they only define inline(always)
// functions, which always get lazy codegen.

mod inline {

    //~ MONO_ITEM fn inline::inlined_function @@ local_transitive_inlining-indirect_user[Internal]
    #[inline(always)]
    pub fn inlined_function() {}
}

pub mod direct_user {
    use super::inline;

    //~ MONO_ITEM fn direct_user::foo @@ local_transitive_inlining-indirect_user[Internal]
    #[inline(always)]
    pub fn foo() {
        inline::inlined_function();
    }
}

pub mod indirect_user {
    use super::direct_user;

    //~ MONO_ITEM fn indirect_user::bar @@ local_transitive_inlining-indirect_user[External]
    pub fn bar() {
        direct_user::foo();
    }
}

pub mod non_user {

    //~ MONO_ITEM fn non_user::baz @@ local_transitive_inlining-non_user[External]
    pub fn baz() {}
}
