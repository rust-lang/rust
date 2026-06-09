// Make sure that indexing an array is only valid with a `usize`, not any other
// integral type.

fn main() {
    fn bar<T>(_: T) {}
    [0][0u8]; //~ ERROR: the type `[{integer}]` cannot be indexed by `u8`

    [0][0]; // should infer to be a usize

    let i = 0;      // i is an IntVar
    [0][i];         // i should be locked to usize
    bar::<isize>(i);  // i should not be re-coerced back to an isize
    //~^ ERROR: mismatched types
}
