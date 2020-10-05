#![feature(const_generics)] //~ WARN the feature `const_generics` is incomplete

// It depends on how we normalize constants and how const equate works if this
// compiles.
//
// Please ping @lcnr if the output if this test changes.


fn bind<const N: usize>(value: [u8; N + 2]) -> [u8; N * 2] {
    //~^ ERROR constant expression depends on a generic parameter
    //~| ERROR constant expression depends on a generic parameter
    todo!()
}

fn main() {
    let mut arr = Default::default();
    arr = bind::<2>(arr);
}
