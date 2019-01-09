// ignore-tidy-linelength
// compile-flags:-Zprint-mono-items=eager
// compile-flags:-Zinline-in-all-cgus

#![deny(dead_code)]
#![feature(start)]

//~ MONO_ITEM fn core::ptr[0]::real_drop_in_place[0]<transitive_drop_glue::Root[0]> @@ transitive_drop_glue-cgu.0[Internal]
struct Root(Intermediate);
//~ MONO_ITEM fn core::ptr[0]::real_drop_in_place[0]<transitive_drop_glue::Intermediate[0]> @@ transitive_drop_glue-cgu.0[Internal]
struct Intermediate(Leaf);
//~ MONO_ITEM fn core::ptr[0]::real_drop_in_place[0]<transitive_drop_glue::Leaf[0]> @@ transitive_drop_glue-cgu.0[Internal]
struct Leaf;

impl Drop for Leaf {
    //~ MONO_ITEM fn transitive_drop_glue::{{impl}}[0]::drop[0]
    fn drop(&mut self) {}
}

struct RootGen<T>(IntermediateGen<T>);
struct IntermediateGen<T>(LeafGen<T>);
struct LeafGen<T>(T);

impl<T> Drop for LeafGen<T> {
    fn drop(&mut self) {}
}

//~ MONO_ITEM fn transitive_drop_glue::start[0]
#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    let _ = Root(Intermediate(Leaf));

    //~ MONO_ITEM fn core::ptr[0]::real_drop_in_place[0]<transitive_drop_glue::RootGen[0]<u32>> @@ transitive_drop_glue-cgu.0[Internal]
    //~ MONO_ITEM fn core::ptr[0]::real_drop_in_place[0]<transitive_drop_glue::IntermediateGen[0]<u32>> @@ transitive_drop_glue-cgu.0[Internal]
    //~ MONO_ITEM fn core::ptr[0]::real_drop_in_place[0]<transitive_drop_glue::LeafGen[0]<u32>> @@ transitive_drop_glue-cgu.0[Internal]
    //~ MONO_ITEM fn transitive_drop_glue::{{impl}}[1]::drop[0]<u32>
    let _ = RootGen(IntermediateGen(LeafGen(0u32)));

    //~ MONO_ITEM fn core::ptr[0]::real_drop_in_place[0]<transitive_drop_glue::RootGen[0]<i16>> @@ transitive_drop_glue-cgu.0[Internal]
    //~ MONO_ITEM fn core::ptr[0]::real_drop_in_place[0]<transitive_drop_glue::IntermediateGen[0]<i16>> @@ transitive_drop_glue-cgu.0[Internal]
    //~ MONO_ITEM fn core::ptr[0]::real_drop_in_place[0]<transitive_drop_glue::LeafGen[0]<i16>> @@ transitive_drop_glue-cgu.0[Internal]
    //~ MONO_ITEM fn transitive_drop_glue::{{impl}}[1]::drop[0]<i16>
    let _ = RootGen(IntermediateGen(LeafGen(0i16)));

    0
}
