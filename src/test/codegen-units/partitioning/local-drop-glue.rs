//
// We specify incremental here because we want to test the partitioning for
// incremental compilation
// We specify opt-level=0 because `drop_in_place` is `Internal` when optimizing
// incremental
// compile-flags:-Zprint-mono-items=lazy
// compile-flags:-Zinline-in-all-cgus -Copt-level=0

#![allow(dead_code)]
#![crate_type = "rlib"]

//~ MONO_ITEM fn std::ptr::drop_in_place::<Struct> - shim(Some(Struct)) @@ local_drop_glue-fallback.cgu[External]
struct Struct {
    _a: u32,
}

impl Drop for Struct {
    //~ MONO_ITEM fn <Struct as std::ops::Drop>::drop @@ local_drop_glue-fallback.cgu[External]
    fn drop(&mut self) {}
}

//~ MONO_ITEM fn std::ptr::drop_in_place::<Outer> - shim(Some(Outer)) @@ local_drop_glue-fallback.cgu[External]
struct Outer {
    _a: Struct,
}

//~ MONO_ITEM fn user @@ local_drop_glue[External]
pub fn user() {
    let _ = Outer { _a: Struct { _a: 0 } };
}

pub mod mod1 {
    use super::Struct;

    //~ MONO_ITEM fn std::ptr::drop_in_place::<mod1::Struct2> - shim(Some(mod1::Struct2)) @@ local_drop_glue-fallback.cgu[External]
    struct Struct2 {
        _a: Struct,
        //~ MONO_ITEM fn std::ptr::drop_in_place::<(u32, Struct)> - shim(Some((u32, Struct))) @@ local_drop_glue-fallback.cgu[Internal]
        _b: (u32, Struct),
    }

    //~ MONO_ITEM fn mod1::user @@ local_drop_glue-mod1[External]
    pub fn user() {
        let _ = Struct2 { _a: Struct { _a: 0 }, _b: (0, Struct { _a: 0 }) };
    }
}
