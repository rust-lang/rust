// Check that a constant-evaluation underflow highlights the correct
// spot (where the underflow occurred).

const ONE: usize = 1;
const TWO: usize = 2;

fn main() {
    let a: [i8; ONE - TWO] = unimplemented!();
    //~^ ERROR could not evaluate constant expression
    //~| attempt to subtract with overflow
}
