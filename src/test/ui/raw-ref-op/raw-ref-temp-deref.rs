// FIXME(#64490) This should be check-pass
// Check that taking the address of a place that contains a dereference is
// allowed.
#![feature(raw_ref_op, type_ascription)]

const PAIR_REF: &(i32, i64) = &(1, 2);

const ARRAY_REF: &[i32; 2] = &[3, 4];
const SLICE_REF: &[i32] = &[5, 6];

fn main() {
    // These are all OK, we're not taking the address of the temporary
    let deref_ref = &raw const *PAIR_REF;                       //~ ERROR not yet implemented
    let field_deref_ref = &raw const PAIR_REF.0;                //~ ERROR not yet implemented
    let deref_ref = &raw const *ARRAY_REF;                      //~ ERROR not yet implemented
    let index_deref_ref = &raw const ARRAY_REF[0];              //~ ERROR not yet implemented
    let deref_ref = &raw const *SLICE_REF;                      //~ ERROR not yet implemented
    let index_deref_ref = &raw const SLICE_REF[1];              //~ ERROR not yet implemented

    let x = 0;
    let ascribe_ref = &raw const (x: i32);                      //~ ERROR not yet implemented
    let ascribe_deref = &raw const (*ARRAY_REF: [i32; 2]);      //~ ERROR not yet implemented
    let ascribe_index_deref = &raw const (ARRAY_REF[0]: i32);   //~ ERROR not yet implemented
}
