// Check that a constant-evaluation underflow highlights the correct
// spot (where the underflow occurred), while also providing the
// overall context for what caused the evaluation.

//@ revisions: old next
//@[next] compile-flags: -Znext-solver

const ONE: usize = 1;
const TWO: usize = 2;
const LEN: usize = ONE - TWO;
//~^ NOTE failed here
//~| ERROR attempt to compute `1_usize - 2_usize`, which would overflow

fn main() {
    let a: [i8; LEN] = unimplemented!();
    //~^ NOTE constant
}
