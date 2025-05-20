//@ compile-flags:-Clink-dead-code
//@ compile-flags: -O

#![deny(dead_code)]
#![crate_type = "lib"]

//~ MONO_ITEM fn std::ptr::drop_in_place::<StructWithDrop> - shim(Some(StructWithDrop)) @@ non_generic_drop_glue-cgu.0[Internal]
struct StructWithDrop {
    x: i32,
}

impl Drop for StructWithDrop {
    //~ MONO_ITEM fn <StructWithDrop as std::ops::Drop>::drop
    fn drop(&mut self) {}
}

struct StructNoDrop {
    x: i32,
}

//~ MONO_ITEM fn std::ptr::drop_in_place::<EnumWithDrop> - shim(Some(EnumWithDrop)) @@ non_generic_drop_glue-cgu.0[Internal]
enum EnumWithDrop {
    A(i32),
}

impl Drop for EnumWithDrop {
    //~ MONO_ITEM fn <EnumWithDrop as std::ops::Drop>::drop
    fn drop(&mut self) {}
}

enum EnumNoDrop {
    A(i32),
}

//~ MONO_ITEM fn start
#[no_mangle]
pub fn start(_: isize, _: *const *const u8) -> isize {
    let _ = StructWithDrop { x: 0 }.x;
    let _ = StructNoDrop { x: 0 }.x;
    let _ = match EnumWithDrop::A(0) {
        EnumWithDrop::A(x) => x,
    };
    let _ = match EnumNoDrop::A(0) {
        EnumNoDrop::A(x) => x,
    };

    0
}
