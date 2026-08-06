// rustfmt-struct_field_align_threshold: 5

#[derive(Fail, Debug, Clone)]
pub enum BuildError {
    LineTooLong { length: usize, limit: usize },
    DisallowedByte { b: u8, pos: usize },
    ContainsNewLine { pos: usize },
}

enum Foo {
    A { a: usize, bbbbb: () },
    B { a: (), bbbbb: () },
}
