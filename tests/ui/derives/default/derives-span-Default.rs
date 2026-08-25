//! Check that all the derives have spans that point to the fields,
//! rather than the #[derive(Default)] line.

struct Error;

#[derive(Default)]
struct Struct {
    x: Error, //~ ERROR
}

#[derive(Default)]
struct TupleStruct(
    Error, //~ ERROR
);

fn main() {}
