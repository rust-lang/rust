//@ check-pass
// Check that taking the address of a place that contains a dereference is
// allowed.
#![feature(type_ascription)]

const PAIR_REF: &(i32, i64) = &(1, 2);

const ARRAY_REF: &[i32; 2] = &[3, 4];
const SLICE_REF: &[i32] = &[5, 6];

fn main() {
    // These are all OK, we're not taking the address of the temporary
    let deref_ref = &raw const *PAIR_REF;
    let field_deref_ref = &raw const PAIR_REF.0;
    let deref_ref = &raw const *ARRAY_REF;
    let index_deref_ref = &raw const ARRAY_REF[0];
    let deref_ref = &raw const *SLICE_REF;
    let index_deref_ref = &raw const SLICE_REF[1];

    let x = 0;
    let ascribe_ref = &raw const type_ascribe!(x, i32);
    let ascribe_deref = &raw const type_ascribe!(*ARRAY_REF, [i32; 2]);
    let ascribe_index_deref = &raw const type_ascribe!(ARRAY_REF[0], i32);
}
