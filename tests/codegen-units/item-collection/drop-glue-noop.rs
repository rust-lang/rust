//@ compile-flags:-Clink-dead-code -Zmir-opt-level=0

#![deny(dead_code)]
#![crate_type = "lib"]

//~ MONO_ITEM fn start
#[no_mangle]
pub fn start(_: isize, _: *const *const u8) -> isize {
    // No item produced for this, it's a no-op drop and so is removed.
    unsafe {
        std::ptr::drop_in_place::<u32>(&mut 0);
    }

    // No choice but to codegen for indirect drop as a function pointer, since we have to produce a
    // function with the right signature. In vtables we can avoid that (tested in
    // instantiation-through-vtable.rs) because we special case null pointer for drop glue since
    // #122662.
    //
    //~ MONO_ITEM fn std::ptr::drop_in_place::<u64> - shim(None) @@ drop_glue_noop-cgu.0[External]
    std::ptr::drop_in_place::<u64> as unsafe fn(*mut u64);

    0
}
