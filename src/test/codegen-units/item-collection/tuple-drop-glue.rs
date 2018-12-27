// ignore-tidy-linelength
// compile-flags:-Zprint-mono-items=eager
// compile-flags:-Zinline-in-all-cgus

#![deny(dead_code)]
#![feature(start)]

//~ MONO_ITEM fn core::ptr[0]::real_drop_in_place[0]<tuple_drop_glue::Dropped[0]> @@ tuple_drop_glue-cgu.0[Internal]
struct Dropped;

impl Drop for Dropped {
    //~ MONO_ITEM fn tuple_drop_glue::{{impl}}[0]::drop[0]
    fn drop(&mut self) {}
}

//~ MONO_ITEM fn tuple_drop_glue::start[0]
#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    //~ MONO_ITEM fn core::ptr[0]::real_drop_in_place[0]<(u32, tuple_drop_glue::Dropped[0])> @@ tuple_drop_glue-cgu.0[Internal]
    let x = (0u32, Dropped);

    //~ MONO_ITEM fn core::ptr[0]::real_drop_in_place[0]<(i16, (tuple_drop_glue::Dropped[0], bool))> @@ tuple_drop_glue-cgu.0[Internal]
    //~ MONO_ITEM fn core::ptr[0]::real_drop_in_place[0]<(tuple_drop_glue::Dropped[0], bool)> @@ tuple_drop_glue-cgu.0[Internal]
    let x = (0i16, (Dropped, true));

    0
}
