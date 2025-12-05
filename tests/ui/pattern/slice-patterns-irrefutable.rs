// Test that we infer the expected type of a pattern to an array of the given length.

#![allow(unused_variables)]

use std::array::TryFromSliceError;
use std::convert::TryInto;

struct Zeroes;
impl Into<[usize; 2]> for Zeroes {
    fn into(self) -> [usize; 2] {
        [0; 2]
    }
}
impl Into<[usize; 3]> for Zeroes {
    fn into(self) -> [usize; 3] {
        [0; 3]
    }
}
impl Into<[usize; 4]> for Zeroes {
    fn into(self) -> [usize; 4] {
        [0; 4]
    }
}

fn zeroes_into() {
    let [a, b, c] = Zeroes.into();
    let [d, e, f]: [_; 3] = Zeroes.into();
}

fn array_try_from(x: &[usize]) -> Result<usize, TryFromSliceError> {
    let [a, b] = x.try_into()?;
    Ok(a + b)
}

fn destructuring_assignment() {
    let a: i32;
    let b;
    [a, b] = Default::default();
}

fn test_nested_array() {
    let a: [_; 3];
    let b;
    //~^ ERROR type annotations needed
    [a, b] = Default::default();
}

fn test_nested_array_type_hint() {
    let a: [_; 3];
    let b;
    [a, b] = Default::default();
    let _: i32 = b[1];
}

fn test_working_nested_array() {
    let a: i32;
    [[a, _, _], _, _] = Default::default();
}

struct Foo<T>([T; 2]);

impl<T: Default + Copy> Default for Foo<T> {
    fn default() -> Self {
        Foo([Default::default(); 2])
    }
}

fn field_array() {
    let a: i32;
    let b;
    Foo([a, b]) = Default::default();
}

fn main() {}
