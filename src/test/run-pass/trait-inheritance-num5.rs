use cmp::{Eq, Ord};
use num::from_int;

pub trait NumExt: Eq, Num {}

pub impl f32: NumExt {}
pub impl int: NumExt {}

fn num_eq_one<T:NumExt>() -> T {
    from_int(1)
}

fn main() {
    num_eq_one::<int>(); // you need to actually use the function to trigger the ICE
}
