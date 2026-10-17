//! Check that all the derives have spans that point to the fields,
//! rather than the #[derive(Hash)] line.

struct Error;

#[derive(Hash)]
enum EnumStructVariant {
    A {
        x: Error, //~ ERROR
    },
}

#[derive(Hash)]
enum EnumTupleVariant {
    A(
        Error, //~ ERROR
    ),
}

#[derive(Hash)]
struct Struct {
    x: Error, //~ ERROR
}

#[derive(Hash)]
struct TupleStruct(
    Error, //~ ERROR
);

fn main() {}
