//@ compile-flags: -Cdebuginfo=2 --crate-type=lib
//@ build-pass
#![feature(adt_const_params)]

const N_ISLANDS: usize = 4;

pub type Matrix = [[usize; N_ISLANDS]; N_ISLANDS];

const EMPTY_MATRIX: Matrix = [[0; N_ISLANDS]; N_ISLANDS];

const fn to_matrix() -> Matrix {
    EMPTY_MATRIX
}

const BRIDGE_MATRIX: [[usize; N_ISLANDS]; N_ISLANDS] = to_matrix();

pub struct Walk<const CURRENT: usize, const REMAINING: Matrix> {
    _p: (),
}

impl Walk<0, BRIDGE_MATRIX> {
    pub const fn new() -> Self {
        Self { _p: () }
    }
}
