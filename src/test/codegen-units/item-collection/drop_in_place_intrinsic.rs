// ignore-tidy-linelength
// compile-flags:-Zprint-mono-items=eager
// compile-flags:-Zinline-in-all-cgus

#![feature(start)]

//~ MONO_ITEM fn core::ptr[0]::real_drop_in_place[0]<drop_in_place_intrinsic::StructWithDtor[0]> @@ drop_in_place_intrinsic-cgu.0[Internal]
struct StructWithDtor(u32);

impl Drop for StructWithDtor {
    //~ MONO_ITEM fn drop_in_place_intrinsic::{{impl}}[0]::drop[0]
    fn drop(&mut self) {}
}

//~ MONO_ITEM fn drop_in_place_intrinsic::start[0]
#[start]
fn start(_: isize, _: *const *const u8) -> isize {

    //~ MONO_ITEM fn core::ptr[0]::real_drop_in_place[0]<[drop_in_place_intrinsic::StructWithDtor[0]; 2]> @@ drop_in_place_intrinsic-cgu.0[Internal]
    let x = [StructWithDtor(0), StructWithDtor(1)];

    drop_slice_in_place(&x);

    0
}

//~ MONO_ITEM fn drop_in_place_intrinsic::drop_slice_in_place[0]
fn drop_slice_in_place(x: &[StructWithDtor]) {
    unsafe {
        // This is the interesting thing in this test case: Normally we would
        // not have drop-glue for the unsized [StructWithDtor]. This has to be
        // generated though when the drop_in_place() intrinsic is used.
        //~ MONO_ITEM fn core::ptr[0]::drop_in_place[0]<[drop_in_place_intrinsic::StructWithDtor[0]]> @@ drop_in_place_intrinsic-cgu.0[Internal]
        //~ MONO_ITEM fn core::ptr[0]::real_drop_in_place[0]<[drop_in_place_intrinsic::StructWithDtor[0]]> @@ drop_in_place_intrinsic-cgu.0[Internal]
        ::std::ptr::drop_in_place(x as *const _ as *mut [StructWithDtor]);
    }
}
