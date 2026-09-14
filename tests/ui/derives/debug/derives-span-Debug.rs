//! Check that all the derives have spans that point to the fields,
//! rather than the #[derive(Debug)] line.

struct Error;

#[derive(Debug)]
enum EnumStructVariant {
    A {
        x: Error, //~ ERROR
    },
}

#[derive(Debug)]
enum EnumTupleVariant {
    A(
        Error, //~ ERROR
    ),
}
#[derive(Debug)]
struct Struct {
    x: Error, //~ ERROR
}

#[derive(Debug)]
struct TupleStruct(
    Error, //~ ERROR
);

fn main() {}
