#![feature(adt_const_params)]

fn func<const V: [u32]>() {}
//~^ ERROR use of unstable library feature `unsized_const_params`

const VALUE: [u32] = [0; 4];
//~^ ERROR mismatched types
//~| ERROR the size for values of type `[u32]` cannot be known at compilation time

fn main() {
    func::<VALUE>();
    //~^ ERROR the size for values of type `[u32]` cannot be known at compilation time
}
