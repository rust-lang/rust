#![deny(dead_code)]

#[derive(Clone)]
enum Enum {
    Variant1, //~ ERROR: variant `Variant1` is never constructed
    Variant2,
}

#[derive(Debug)]
enum TupleVariant {
    Variant1(i32), //~ ERROR: variant `Variant1` is never constructed
    Variant2,
}

#[derive(Debug)]
enum StructVariant {
    Variant1 { id: i32 }, //~ ERROR: variant `Variant1` is never constructed
    Variant2,
}

fn main() {
    let e = Enum::Variant2;
    e.clone();

    let _ = TupleVariant::Variant2;
    let _ = StructVariant::Variant2;
}
