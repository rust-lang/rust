// Ensure that we don't allow taking the address of temporary values
#![feature(raw_ref_op)]

const PAIR: (i32, i64) = (1, 2);
const PAIR_REF: &(i32, i64) = &(1, 2);

const ARRAY: [i32; 2] = [1, 2];
const ARRAY_REF: &[i32; 2] = &[3, 4];
const SLICE_REF: &[i32] = &[5, 6];

fn main() {
    // These are all OK, we're not taking the address of the temporary
    let deref_ref = &raw const *PAIR_REF;               //~ ERROR not yet implemented
    let field_deref_ref = &raw const PAIR_REF.0;        //~ ERROR not yet implemented
    let deref_ref = &raw const *ARRAY_REF;              //~ ERROR not yet implemented
    let field_deref_ref = &raw const ARRAY_REF[0];      //~ ERROR not yet implemented
    let deref_ref = &raw const *SLICE_REF;              //~ ERROR not yet implemented
    let field_deref_ref = &raw const SLICE_REF[1];      //~ ERROR not yet implemented
}
