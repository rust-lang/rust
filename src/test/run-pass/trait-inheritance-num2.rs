// A more complex example of numeric extensions

use cmp::{Eq, Ord};
use num::from_int;

extern mod std;
use std::cmp::FuzzyEq;

pub trait TypeExt {}


pub impl u8: TypeExt {}
pub impl u16: TypeExt {}
pub impl u32: TypeExt {}
pub impl u64: TypeExt {}
pub impl uint: TypeExt {}

pub impl i8: TypeExt {}
pub impl i16: TypeExt {}
pub impl i32: TypeExt {}
pub impl i64: TypeExt {}
pub impl int: TypeExt {}

pub impl f32: TypeExt {}
pub impl f64: TypeExt {}
pub impl float: TypeExt {}


pub trait NumExt: TypeExt Eq Ord Num {}

pub impl u8: NumExt {}
pub impl u16: NumExt {}
pub impl u32: NumExt {}
pub impl u64: NumExt {}
pub impl uint: NumExt {}

pub impl i8: NumExt {}
pub impl i16: NumExt {}
pub impl i32: NumExt {}
pub impl i64: NumExt {}
pub impl int: NumExt {}

pub impl f32: NumExt {}
pub impl f64: NumExt {}
pub impl float: NumExt {}


pub trait UnSignedExt: NumExt {}

pub impl u8: UnSignedExt {}
pub impl u16: UnSignedExt {}
pub impl u32: UnSignedExt {}
pub impl u64: UnSignedExt {}
pub impl uint: UnSignedExt {}


pub trait SignedExt: NumExt {}

pub impl i8: SignedExt {}
pub impl i16: SignedExt {}
pub impl i32: SignedExt {}
pub impl i64: SignedExt {}
pub impl int: SignedExt {}

pub impl f32: SignedExt {}
pub impl f64: SignedExt {}
pub impl float: SignedExt {}


pub trait IntegerExt: NumExt {}

pub impl u8: IntegerExt {}
pub impl u16: IntegerExt {}
pub impl u32: IntegerExt {}
pub impl u64: IntegerExt {}
pub impl uint: IntegerExt {}

pub impl i8: IntegerExt {}
pub impl i16: IntegerExt {}
pub impl i32: IntegerExt {}
pub impl i64: IntegerExt {}
pub impl int: IntegerExt {}


pub trait FloatExt: NumExt FuzzyEq {}

pub impl f32: FloatExt {}
pub impl f64: FloatExt {}
pub impl float: FloatExt {}


fn test_float_ext<T:FloatExt>(n: T) { io::println(fmt!("%?", n < n)) }

fn main() {
    test_float_ext(1f32);
}