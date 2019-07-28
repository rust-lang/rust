// run-pass
#![allow(dead_code)]
#[derive(Debug)]
struct Unit;

#[derive(Debug)]
struct Tuple(isize, usize);

#[derive(Debug)]
struct Struct { x: isize, y: usize }

#[derive(Debug)]
enum Enum {
    Nullary,
    Variant(isize, usize),
    StructVariant { x: isize, y : usize }
}

#[derive(Debug)]
struct Pointers(*const dyn Send, *mut dyn Sync);

macro_rules! t {
    ($x:expr, $expected:expr) => {
        assert_eq!(format!("{:?}", $x), $expected.to_string())
    }
}

pub fn main() {
    t!(Unit, "Unit");
    t!(Tuple(1, 2), "Tuple(1, 2)");
    t!(Struct { x: 1, y: 2 }, "Struct { x: 1, y: 2 }");
    t!(Enum::Nullary, "Nullary");
    t!(Enum::Variant(1, 2), "Variant(1, 2)");
    t!(Enum::StructVariant { x: 1, y: 2 }, "StructVariant { x: 1, y: 2 }");
}
