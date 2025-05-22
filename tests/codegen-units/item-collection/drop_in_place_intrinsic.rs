//@ compile-flags:-Clink-dead-code
//@ compile-flags:-Zinline-mir=no
//@ compile-flags: -O

#![crate_type = "lib"]

//~ MONO_ITEM fn std::ptr::drop_in_place::<StructWithDtor> - shim(Some(StructWithDtor)) @@ drop_in_place_intrinsic-cgu.0[Internal]
struct StructWithDtor(u32);

impl Drop for StructWithDtor {
    //~ MONO_ITEM fn <StructWithDtor as std::ops::Drop>::drop
    fn drop(&mut self) {}
}

//~ MONO_ITEM fn start
#[no_mangle]
pub fn start(_: isize, _: *const *const u8) -> isize {
    //~ MONO_ITEM fn std::ptr::drop_in_place::<[StructWithDtor; 2]> - shim(Some([StructWithDtor; 2])) @@ drop_in_place_intrinsic-cgu.0[Internal]
    let x = [StructWithDtor(0), StructWithDtor(1)];

    drop_slice_in_place(&x);

    0
}

//~ MONO_ITEM fn drop_slice_in_place
fn drop_slice_in_place(x: &[StructWithDtor]) {
    unsafe {
        // This is the interesting thing in this test case: Normally we would
        // not have drop-glue for the unsized [StructWithDtor]. This has to be
        // generated though when the drop_in_place() intrinsic is used.
        //~ MONO_ITEM fn std::ptr::drop_in_place::<[StructWithDtor]> - shim(Some([StructWithDtor])) @@ drop_in_place_intrinsic-cgu.0[Internal]
        ::std::ptr::drop_in_place(x as *const _ as *mut [StructWithDtor]);
    }
}
