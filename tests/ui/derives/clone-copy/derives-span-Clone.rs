//! Check that all the derives have spans that point to the fields,
//! rather than the #[derive(Clone)] line.

struct Error;

#[derive(Clone)]
enum EnumStructVariant {
    A {
        x: Error, //~ ERROR
    },
}

#[derive(Clone)]
enum EnumTupleVariant {
    A(
        Error, //~ ERROR
    ),
}

#[derive(Clone)]
struct Struct {
    x: Error, //~ ERROR
}

#[derive(Clone)]
struct TupleStruct(
    Error, //~ ERROR
);

fn main() {}
