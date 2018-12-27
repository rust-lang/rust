// ignore-tidy-linelength
// compile-flags:-Zprint-mono-items=eager
// compile-flags:-Zinline-in-all-cgus

#![deny(dead_code)]
#![feature(start)]

//~ MONO_ITEM fn core::ptr[0]::real_drop_in_place[0]<non_generic_drop_glue::StructWithDrop[0]> @@ non_generic_drop_glue-cgu.0[Internal]
struct StructWithDrop {
    x: i32
}

impl Drop for StructWithDrop {
    //~ MONO_ITEM fn non_generic_drop_glue::{{impl}}[0]::drop[0]
    fn drop(&mut self) {}
}

struct StructNoDrop {
    x: i32
}

//~ MONO_ITEM fn core::ptr[0]::real_drop_in_place[0]<non_generic_drop_glue::EnumWithDrop[0]> @@ non_generic_drop_glue-cgu.0[Internal]
enum EnumWithDrop {
    A(i32)
}

impl Drop for EnumWithDrop {
    //~ MONO_ITEM fn non_generic_drop_glue::{{impl}}[1]::drop[0]
    fn drop(&mut self) {}
}

enum EnumNoDrop {
    A(i32)
}

//~ MONO_ITEM fn non_generic_drop_glue::start[0]
#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    let _ = StructWithDrop { x: 0 }.x;
    let _ = StructNoDrop { x: 0 }.x;
    let _ = match EnumWithDrop::A(0) {
        EnumWithDrop::A(x) => x
    };
    let _ = match EnumNoDrop::A(0) {
        EnumNoDrop::A(x) => x
    };

    0
}
