//! Check that all the derives have spans that point to the fields,
//! rather than the #[derive(PartialOrd)] line.

#[derive(PartialEq)]
struct Error;

#[derive(PartialOrd, PartialEq)]
enum EnumStructVariant {
    A {
        x: Error, //~ ERROR
    },
}

#[derive(PartialOrd, PartialEq)]
enum EnumTupleVariant {
    A(
        Error, //~ ERROR
    ),
}

#[derive(PartialOrd, PartialEq)]
struct Struct {
    x: Error, //~ ERROR
}

#[derive(PartialOrd, PartialEq)]
struct TupleStruct(
    Error, //~ ERROR
);

fn main() {}
