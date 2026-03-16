//! Check that all the derives have spans that point to the fields,
//! rather than the #[derive(Eq)] line.

#[derive(PartialEq)]
struct Error;

#[derive(Eq, PartialEq)]
enum EnumStructVariant {
    A {
        x: Error, //~ ERROR
    },
}

#[derive(Eq, PartialEq)]
enum EnumTupleVariant {
    A(
        Error, //~ ERROR
    ),
}

#[derive(Eq, PartialEq)]
struct Struct {
    x: Error, //~ ERROR
}

#[derive(Eq, PartialEq)]
struct TupleStruct(
    Error, //~ ERROR
);

fn main() {}
