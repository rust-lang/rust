//! Check that all the derives have spans that point to the fields,
//! rather than the #[derive(PartialEq)] line.

struct Error;

#[derive(PartialEq)]
enum EnumStructVariant {
    A {
        x: Error, //~ ERROR
    },
}

#[derive(PartialEq)]
enum EnumTupleVariant {
    A(
        Error, //~ ERROR
    ),
}
#[derive(PartialEq)]
struct Struct {
    x: Error, //~ ERROR
}

#[derive(PartialEq)]
struct TupleStruct(
    Error, //~ ERROR
);

fn main() {}
