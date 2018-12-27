// ignore-tidy-linelength
// We specify -Z incremental here because we want to test the partitioning for
// incremental compilation
// compile-flags:-Zprint-mono-items=lazy -Zincremental=tmp/partitioning-tests/inlining-from-extern-crate
// compile-flags:-Zinline-in-all-cgus

#![crate_type="lib"]

// aux-build:cgu_explicit_inlining.rs
extern crate cgu_explicit_inlining;

// This test makes sure that items inlined from external crates are privately
// instantiated in every codegen unit they are used in.

//~ MONO_ITEM fn cgu_explicit_inlining::inlined[0] @@ inlining_from_extern_crate[Internal] inlining_from_extern_crate-mod1[Internal]
//~ MONO_ITEM fn cgu_explicit_inlining::always_inlined[0] @@ inlining_from_extern_crate[Internal] inlining_from_extern_crate-mod2[Internal]

//~ MONO_ITEM fn inlining_from_extern_crate::user[0] @@ inlining_from_extern_crate[External]
pub fn user()
{
    cgu_explicit_inlining::inlined();
    cgu_explicit_inlining::always_inlined();

    // does not generate a monomorphization in this crate
    cgu_explicit_inlining::never_inlined();
}

pub mod mod1 {
    use cgu_explicit_inlining;

    //~ MONO_ITEM fn inlining_from_extern_crate::mod1[0]::user[0] @@ inlining_from_extern_crate-mod1[External]
    pub fn user()
    {
        cgu_explicit_inlining::inlined();

        // does not generate a monomorphization in this crate
        cgu_explicit_inlining::never_inlined();
    }
}

pub mod mod2 {
    use cgu_explicit_inlining;

    //~ MONO_ITEM fn inlining_from_extern_crate::mod2[0]::user[0] @@ inlining_from_extern_crate-mod2[External]
    pub fn user()
    {
        cgu_explicit_inlining::always_inlined();

        // does not generate a monomorphization in this crate
        cgu_explicit_inlining::never_inlined();
    }
}
