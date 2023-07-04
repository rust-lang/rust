// Check that a constant-evaluation underflow highlights the correct
// spot (where the underflow occurred), while also providing the
// overall context for what caused the evaluation.

// revisions: old next
//[next] compile-flags: -Ztrait-solver=next

const ONE: usize = 1;
const TWO: usize = 2;
const LEN: usize = ONE - TWO;
//~^ ERROR constant

fn main() {
    let a: [i8; LEN] = unimplemented!();
//~^ constant
}
