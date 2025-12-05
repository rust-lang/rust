//@ known-bug: #137582
#![feature(adt_const_params)]

mod lib {
    pub type Matrix = [&'static u32];

    const EMPTY_MATRIX: Matrix = [[0; 4]; 4];

    pub struct Walk<const CURRENT: usize, const REMAINING: Matrix> {
        _p: (),
    }

    impl<const CURRENT: usize> Walk<CURRENT, EMPTY_MATRIX> {}
}

fn main() {}
