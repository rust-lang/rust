//@ check-pass
#![feature(adt_const_params)]

const EMPTY_MATRIX: <Type as Trait>::Matrix = [0; 1];

pub struct Walk<const REMAINING: <Type as Trait>::Matrix> {}

impl Walk<EMPTY_MATRIX> {
    pub const fn new() -> Self {
        Self {}
    }
}

pub enum Type {}
pub trait Trait {
    type Matrix;
}
impl Trait for Type {
    type Matrix = [usize; 1];
}

fn main() {
    let _ = Walk::new();
}
