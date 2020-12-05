// A version of coerce-expect-unsized that uses type ascription.
// Doesn't work so far, but supposed to work eventually

// build-pass

#![feature(box_syntax, type_ascription)]

use std::fmt::Debug;

pub fn main() {
    let _ = box { [1, 2, 3] }: Box<[i32]>;
    let _ = box if true { [1, 2, 3] } else { [1, 3, 4] }: Box<[i32]>;
    let _ = box match true { true => [1, 2, 3], false => [1, 3, 4] }: Box<[i32]>;
    let _ = box { |x| (x as u8) }: Box<dyn Fn(i32) -> _>;
    let _ = box if true { false } else { true }: Box<dyn Debug>;
    let _ = box match true { true => 'a', false => 'b' }: Box<dyn Debug>;

    let _ = &{ [1, 2, 3] }: &[i32];
    let _ = &if true { [1, 2, 3] } else { [1, 3, 4] }: &[i32];
    let _ = &match true { true => [1, 2, 3], false => [1, 3, 4] }: &[i32];
    let _ = &{ |x| (x as u8) }: &dyn Fn(i32) -> _;
    let _ = &if true { false } else { true }: &dyn Debug;
    let _ = &match true { true => 'a', false => 'b' }: &dyn Debug;

    let _ = Box::new([1, 2, 3]): Box<[i32]>;
    let _ = Box::new(|x| (x as u8)): Box<dyn Fn(i32) -> _>;

    let _ = vec![
        Box::new(|x| (x as u8)),
        box |x| (x as i16 as u8),
    ]: Vec<Box<dyn Fn(i32) -> _>>;
}
