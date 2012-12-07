use cmp::{Eq, Ord};
use num::from_int;

pub trait NumExt: Eq Ord Num {}

pub impl f32: NumExt {}

fn num_eq_one<T:NumExt>(n: T) { io::println(fmt!("%?", n == from_int(1))) }

fn main() {
    num_eq_one(1f32); // you need to actually use the function to trigger the ICE
}