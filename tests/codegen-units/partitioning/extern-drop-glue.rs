//

// We specify incremental here because we want to test the partitioning for
// incremental compilation
// We specify opt-level=0 because `drop_in_place` is `Internal` when optimizing
// incremental
// compile-flags:-Zprint-mono-items=lazy
// compile-flags:-Zinline-in-all-cgus -Copt-level=0

#![allow(dead_code)]
#![crate_type = "rlib"]

// aux-build:cgu_extern_drop_glue.rs
extern crate cgu_extern_drop_glue;

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
