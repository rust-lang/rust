
/// A Doc comment
#[AnAttribute]
pub struct Foo {
    #[rustfmt_skip]
    f :   SomeType, // Comment beside a field
    f: SomeType, // Comment beside a field
    // Comment on a field
    #[AnAttribute]
    g: SomeOtherType,
    /// A doc comment on a field
    h: AThirdType,
}

struct Bar;

// With a where clause and generics.
pub struct Foo<'a, Y: Baz>
    where X: Whatever
{
    f: SomeType, // Comment beside a field
}

struct Baz {
    a: A, // Comment A
    b: B, // Comment B
    c: C, // Comment C
}

struct Baz {
    // Comment A
    a: A,
    // Comment B
    b: B,
    // Comment C
    c: C,
}
