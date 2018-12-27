// ignore-tidy-linelength

// We specify -Z incremental here because we want to test the partitioning for
// incremental compilation
// compile-flags:-Zprint-mono-items=lazy -Zincremental=tmp/partitioning-tests/extern-drop-glue
// compile-flags:-Zinline-in-all-cgus

#![allow(dead_code)]
#![crate_type="rlib"]

// aux-build:cgu_extern_drop_glue.rs
extern crate cgu_extern_drop_glue;

//~ MONO_ITEM fn core::ptr[0]::real_drop_in_place[0]<cgu_extern_drop_glue::Struct[0]> @@ extern_drop_glue[Internal] extern_drop_glue-mod1[Internal]

struct LocalStruct(cgu_extern_drop_glue::Struct);

//~ MONO_ITEM fn extern_drop_glue::user[0] @@ extern_drop_glue[External]
pub fn user()
{
    //~ MONO_ITEM fn core::ptr[0]::real_drop_in_place[0]<extern_drop_glue::LocalStruct[0]> @@ extern_drop_glue[Internal]
    let _ = LocalStruct(cgu_extern_drop_glue::Struct(0));
}

pub mod mod1 {
    use cgu_extern_drop_glue;

    struct LocalStruct(cgu_extern_drop_glue::Struct);

    //~ MONO_ITEM fn extern_drop_glue::mod1[0]::user[0] @@ extern_drop_glue-mod1[External]
    pub fn user()
    {
        //~ MONO_ITEM fn core::ptr[0]::real_drop_in_place[0]<extern_drop_glue::mod1[0]::LocalStruct[0]> @@ extern_drop_glue-mod1[Internal]
        let _ = LocalStruct(cgu_extern_drop_glue::Struct(0));
    }
}
