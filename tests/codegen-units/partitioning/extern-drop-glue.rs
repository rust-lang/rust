//@ incremental
//@ compile-flags: -Copt-level=0

#![crate_type = "rlib"]

//@ aux-build:cgu_extern_drop_glue.rs
extern crate cgu_extern_drop_glue;

// This test checks that drop glue is generated, even for types not defined in this crate, and all
// drop glue is put in the fallback CGU.

//~ MONO_ITEM fn std::ptr::drop_in_place::<cgu_extern_drop_glue::Struct> - shim(Some(cgu_extern_drop_glue::Struct)) @@ extern_drop_glue-fallback.cgu[External]

struct LocalStruct(cgu_extern_drop_glue::Struct);

//~ MONO_ITEM fn user @@ extern_drop_glue[External]
pub fn user() {
    //~ MONO_ITEM fn std::ptr::drop_in_place::<LocalStruct> - shim(Some(LocalStruct)) @@ extern_drop_glue-fallback.cgu[External]
    let _ = LocalStruct(cgu_extern_drop_glue::Struct(0));
}

pub mod mod1 {
    use cgu_extern_drop_glue;

    struct LocalStruct(cgu_extern_drop_glue::Struct);

    //~ MONO_ITEM fn mod1::user @@ extern_drop_glue-mod1[External]
    pub fn user() {
        //~ MONO_ITEM fn std::ptr::drop_in_place::<mod1::LocalStruct> - shim(Some(mod1::LocalStruct)) @@ extern_drop_glue-fallback.cgu[External]
        let _ = LocalStruct(cgu_extern_drop_glue::Struct(0));
    }
}
