//@ compile-flags:-Clink-dead-code
//@ compile-flags: -O

#![deny(dead_code)]
#![crate_type = "lib"]

//~ MONO_ITEM fn std::ptr::drop_in_place::<Dropped> - shim(Some(Dropped)) @@ tuple_drop_glue-cgu.0[Internal]
struct Dropped;

impl Drop for Dropped {
    //~ MONO_ITEM fn <Dropped as std::ops::Drop>::drop
    fn drop(&mut self) {}
}

//~ MONO_ITEM fn start
#[no_mangle]
pub fn start(_: isize, _: *const *const u8) -> isize {
    //~ MONO_ITEM fn std::ptr::drop_in_place::<(u32, Dropped)> - shim(Some((u32, Dropped))) @@ tuple_drop_glue-cgu.0[Internal]
    let x = (0u32, Dropped);

    //~ MONO_ITEM fn std::ptr::drop_in_place::<(i16, (Dropped, bool))> - shim(Some((i16, (Dropped, bool)))) @@ tuple_drop_glue-cgu.0[Internal]
    //~ MONO_ITEM fn std::ptr::drop_in_place::<(Dropped, bool)> - shim(Some((Dropped, bool)))  @@ tuple_drop_glue-cgu.0[Internal]
    let x = (0i16, (Dropped, true));

    0
}
