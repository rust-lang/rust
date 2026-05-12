#![feature(pin_ergonomics)]
// Ensure that we emit either `Drop::drop` or `Drop::pin_drop` for types that implement `Drop`.

//@ compile-flags:-Clink-dead-code
//@ compile-flags:--crate-type=lib
//@ rustc-env:MONO_TEST=1

//~ MONO_ITEM fn std::ptr::drop_glue::<StructWithDrop> - shim(Some(StructWithDrop))
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

//~ MONO_ITEM fn std::ptr::drop_glue::<StructWithPinDrop> - shim(Some(StructWithPinDrop))
struct StructWithPinDrop {
    x: i32,
}

impl Drop for StructWithPinDrop {
    //~ MONO_ITEM fn <StructWithPinDrop as std::ops::Drop>::pin_drop
    fn pin_drop(&pin mut self) {}
}

//~ MONO_ITEM fn std::ptr::drop_glue::<StructPinV2WithPinDrop> - shim(Some(StructPinV2WithPinDrop))
#[pin_v2]
struct StructPinV2WithPinDrop {
    x: i32,
}

impl Drop for StructPinV2WithPinDrop {
    //~ MONO_ITEM fn <StructPinV2WithPinDrop as std::ops::Drop>::pin_drop
    fn pin_drop(&pin mut self) {}
}
