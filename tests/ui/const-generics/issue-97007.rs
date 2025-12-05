//@ check-pass

#![feature(adt_const_params, generic_const_exprs)]
#![allow(incomplete_features)]

mod lib {
    const N_ISLANDS: usize = 4;
    const N_BRIDGES: usize = 7;
    const BRIDGES: [(usize, usize); 7] = [(0, 1), (0, 1), (0, 2), (0, 3), (0, 3), (1, 2), (2, 3)];

    pub type Matrix = [[usize; N_ISLANDS]; N_ISLANDS];

    const EMPTY_MATRIX: Matrix = [[0; N_ISLANDS]; N_ISLANDS];

    const fn build(mut matrix: Matrix, (to, from): (usize, usize)) -> Matrix {
        matrix[to][from] += 1;
        matrix[from][to] += 1;
        matrix
    }

    pub const fn walk(mut matrix: Matrix, from: usize, to: usize) -> Matrix {
        matrix[from][to] -= 1;
        matrix[to][from] -= 1;
        matrix
    }

    const fn to_matrix(bridges: [(usize, usize); N_BRIDGES]) -> Matrix {
        let matrix = EMPTY_MATRIX;

        let matrix = build(matrix, bridges[0]);
        let matrix = build(matrix, bridges[1]);
        let matrix = build(matrix, bridges[2]);
        let matrix = build(matrix, bridges[3]);
        let matrix = build(matrix, bridges[4]);
        let matrix = build(matrix, bridges[5]);
        let matrix = build(matrix, bridges[6]);

        matrix
    }

    const BRIDGE_MATRIX: [[usize; N_ISLANDS]; N_ISLANDS] = to_matrix(BRIDGES);

    pub struct Walk<const CURRENT: usize, const REMAINING: Matrix> {
        _p: (),
    }

    impl Walk<0, BRIDGE_MATRIX> {
        pub const fn new() -> Self {
            Self { _p: () }
        }
    }

    impl<const CURRENT: usize, const REMAINING: Matrix> Walk<CURRENT, REMAINING> {
        pub fn proceed_to<const NEXT: usize>(
            self,
        ) -> Walk<NEXT, { walk(REMAINING, CURRENT, NEXT) }> {
            Walk { _p: () }
        }
    }

    pub struct Trophy {
        _p: (),
    }

    impl<const CURRENT: usize> Walk<CURRENT, EMPTY_MATRIX> {
        pub fn collect_prize(self) -> Trophy {
            Trophy { _p: () }
        }
    }
}

pub use lib::{Trophy, Walk};

fn main() {
    // Example, taking the first step
    let _ = Walk::new().proceed_to::<1>();

    // Don't be so eager to collect the trophy
    // let trophy = Walk::new()
    //     .proceed_to::<1>()
    //     .proceed_to::<0>()
    //     .collect_prize();

    // Can't just make a Trophy out of thin air, you must earn it
    // let trophy: Trophy = Trophy { _p: () };

    // Can you collect the Trophy?
}
