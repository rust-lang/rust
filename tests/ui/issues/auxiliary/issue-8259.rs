pub enum Foo<'a> {
    A,
    B(&'a str),
}
