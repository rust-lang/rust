//@ compile-flags:-Clink-dead-code
//@ compile-flags: -O

#![deny(dead_code)]
#![crate_type = "lib"]

//~ MONO_ITEM fn std::ptr::drop_in_place::<Root> - shim(Some(Root)) @@ transitive_drop_glue-cgu.0[Internal]
struct Root(#[allow(dead_code)] Intermediate);
//~ MONO_ITEM fn std::ptr::drop_in_place::<Intermediate> - shim(Some(Intermediate)) @@ transitive_drop_glue-cgu.0[Internal]
struct Intermediate(#[allow(dead_code)] Leaf);
//~ MONO_ITEM fn std::ptr::drop_in_place::<Leaf> - shim(Some(Leaf)) @@ transitive_drop_glue-cgu.0[Internal]
struct Leaf;

impl Drop for Leaf {
    //~ MONO_ITEM fn <Leaf as std::ops::Drop>::drop
    fn drop(&mut self) {}
}

struct RootGen<T>(#[allow(dead_code)] IntermediateGen<T>);
struct IntermediateGen<T>(#[allow(dead_code)] LeafGen<T>);
struct LeafGen<T>(#[allow(dead_code)] T);

impl<T> Drop for LeafGen<T> {
    fn drop(&mut self) {}
}

//~ MONO_ITEM fn start
#[no_mangle]
pub fn start(_: isize, _: *const *const u8) -> isize {
    let _ = Root(Intermediate(Leaf));

    //~ MONO_ITEM fn std::ptr::drop_in_place::<RootGen<u32>> - shim(Some(RootGen<u32>)) @@ transitive_drop_glue-cgu.0[Internal]
    //~ MONO_ITEM fn std::ptr::drop_in_place::<IntermediateGen<u32>> - shim(Some(IntermediateGen<u32>)) @@ transitive_drop_glue-cgu.0[Internal]
    //~ MONO_ITEM fn std::ptr::drop_in_place::<LeafGen<u32>> - shim(Some(LeafGen<u32>)) @@ transitive_drop_glue-cgu.0[Internal]
    //~ MONO_ITEM fn <LeafGen<u32> as std::ops::Drop>::drop
    let _ = RootGen(IntermediateGen(LeafGen(0u32)));

    //~ MONO_ITEM fn std::ptr::drop_in_place::<RootGen<i16>> - shim(Some(RootGen<i16>)) @@ transitive_drop_glue-cgu.0[Internal]
    //~ MONO_ITEM fn std::ptr::drop_in_place::<IntermediateGen<i16>> - shim(Some(IntermediateGen<i16>)) @@ transitive_drop_glue-cgu.0[Internal]
    //~ MONO_ITEM fn std::ptr::drop_in_place::<LeafGen<i16>> - shim(Some(LeafGen<i16>)) @@ transitive_drop_glue-cgu.0[Internal]
    //~ MONO_ITEM fn <LeafGen<i16> as std::ops::Drop>::drop
    let _ = RootGen(IntermediateGen(LeafGen(0i16)));

    0
}
