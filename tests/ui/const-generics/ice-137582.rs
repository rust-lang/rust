#![feature(adt_const_params)]

mod lib {
    pub type Matrix = [&'static u32];

    const EMPTY_MATRIX: Matrix = [[0; 4]; 4];
    //~^ ERROR the size for values of type `[&'static u32]` cannot be known at compilation time
    //~| ERROR mismatched types
    //~| ERROR mismatched types

    pub struct Walk<const CURRENT: usize, const REMAINING: Matrix> {
        //~^ ERROR use of unstable library feature `unsized_const_params`
        _p: (),
    }

    impl<const CURRENT: usize> Walk<CURRENT, EMPTY_MATRIX> {}
    //~^ ERROR the size for values of type `[&'static u32]` cannot be known at compilation time
}

fn main() {}
