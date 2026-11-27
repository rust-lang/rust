//! Test for fake borrows and fake reads emitted as part of lowering nested indexing expressions.
//! If a slice is indexed into, its bounds check needs to hold through subsequent indexing in the
//! same place expression, to prevent out-of-bounds accesses. Currently, that's accomplished by
//! taking shallow fake borrows of pointers above such a slice after its bounds check. These are
//! kept alive by a corresponding fake read following each bounds check thereafter. Having a single
//! fake read for each fake borrow after the final bounds check is insufficient if an index
//! expression diverges; see <https://github.com/rust-lang/rust/issues/161852>.
//@ skip-filecheck

// EMIT_MIR slice_index_fake_reads.three_slices.built.after.mir
fn three_slices<'a, T>(x: &'a [&'a [&'a [T]]]) -> &'a T {
    &x[0][0][0]
}

// EMIT_MIR slice_index_fake_reads.array_first.built.after.mir
fn array_first<'a, T, const N: usize>(x: &'a [&'a [&'a [T]]; N]) -> &'a T {
    &x[0][0][0]
}

// EMIT_MIR slice_index_fake_reads.array_second.built.after.mir
fn array_second<'a, T, const N: usize>(x: &'a [&'a [&'a [T]; N]]) -> &'a T {
    &x[0][0][0]
}

// EMIT_MIR slice_index_fake_reads.array_third.built.after.mir
fn array_third<'a, T, const N: usize>(x: &'a [&'a [&'a [T; N]]]) -> &'a T {
    &x[0][0][0]
}
