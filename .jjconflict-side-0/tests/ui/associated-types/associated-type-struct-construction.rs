// Check that fully qualified syntax can be used in struct expressions in patterns.
// In other words, check that structs can constructed and destructed via an associated type.
//
//@ run-pass

fn main() {
    let <Type as Trait>::Assoc { field } = <Type as Trait>::Assoc { field: 2 };
    assert_eq!(field, 2);
}

struct Struct {
    field: i8,
}

struct Type;

trait Trait {
    type Assoc;
}

impl Trait for Type {
    type Assoc = Struct;
}
