// Check that a constant-evaluation underflow highlights the correct
// spot (where the underflow occurred), while also providing the
// overall context for what caused the evaluation.

const ONE: usize = 1;
const TWO: usize = 2;
const LEN: usize = ONE - TWO;
//~^ ERROR any use of this value will cause an error

fn main() {
    let a: [i8; LEN] = unimplemented!();
//~^ ERROR E0080
}
