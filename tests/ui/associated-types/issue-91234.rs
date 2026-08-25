//@ check-pass

struct Struct;

trait Trait {
    type Type;
}

enum Enum<'a> where &'a Struct: Trait {
    Variant(<&'a Struct as Trait>::Type)
}

fn main() {}
