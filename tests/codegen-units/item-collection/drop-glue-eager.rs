// Ensure that we *eagerly* monomorphize drop instances for structs with lifetimes.

//@ compile-flags:-Clink-dead-code
//@ compile-flags:--crate-type=lib

//~ MONO_ITEM fn std::ptr::drop_in_place::<StructWithDrop> - shim(Some(StructWithDrop))
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

//~ MONO_ITEM fn std::ptr::drop_in_place::<EnumWithDrop> - shim(Some(EnumWithDrop))
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

// We should be able to monomorphize drops for struct with lifetimes.
impl<'a> Drop for StructWithDropAndLt<'a> {
    //~ MONO_ITEM fn <StructWithDropAndLt<'_> as std::ops::Drop>::drop
    fn drop(&mut self) {}
}

//~ MONO_ITEM fn std::ptr::drop_in_place::<StructWithDropAndLt<'_>> - shim(Some(StructWithDropAndLt<'_>))
struct StructWithDropAndLt<'a> {
    x: &'a i32,
}

// Make sure we don't ICE when checking impossible predicates for the struct.
// Regression test for <https://github.com/rust-lang/rust/issues/135515>.
//~ MONO_ITEM fn std::ptr::drop_in_place::<StructWithLtAndPredicate<'_>> - shim(Some(StructWithLtAndPredicate<'_>))
struct StructWithLtAndPredicate<'a: 'a> {
    x: &'a i32,
}

// We should be able to monomorphize drops for struct with lifetimes.
impl<'a> Drop for StructWithLtAndPredicate<'a> {
    //~ MONO_ITEM fn <StructWithLtAndPredicate<'_> as std::ops::Drop>::drop
    fn drop(&mut self) {}
}
