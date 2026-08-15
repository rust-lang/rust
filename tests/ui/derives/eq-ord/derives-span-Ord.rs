//! Check that all the derives have spans that point to the fields,
//! rather than the #[derive(Ord)] line.

#[derive(Eq, PartialOrd, PartialEq)]
struct Error;

#[derive(Ord, Eq, PartialOrd, PartialEq)]
enum EnumStructVariant {
    A {
        x: Error, //~ ERROR
    },
}

#[derive(Ord, Eq, PartialOrd, PartialEq)]
enum EnumTupleVariant {
    A(
        Error, //~ ERROR
    ),
}

#[derive(Ord, Eq, PartialOrd, PartialEq)]
struct Struct {
    x: Error, //~ ERROR
}

#[derive(Ord, Eq, PartialOrd, PartialEq)]
struct TupleStruct(
    Error, //~ ERROR
);

fn main() {}
