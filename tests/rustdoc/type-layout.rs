// compile-flags: --show-type-layout -Z unstable-options

// @hasraw type_layout/struct.Foo.html 'Size: '
// @hasraw - ' bytes'
// @has - '//*[@id="layout"]/a[@href="#layout"]' ''
pub struct Foo {
    pub a: usize,
    b: Vec<String>,
}

// @hasraw type_layout/enum.Bar.html 'Size: '
// @hasraw - ' bytes'
pub enum Bar<'a> {
    A(String),
    B(&'a str, (std::collections::HashMap<String, usize>, Foo)),
}

// @hasraw type_layout/union.Baz.html 'Size: '
// @hasraw - ' bytes'
pub union Baz {
    a: &'static str,
    b: usize,
    c: &'static [u8],
}

// @hasraw type_layout/struct.X.html 'Size: '
// @hasraw - ' bytes'
pub struct X(usize);

// @hasraw type_layout/struct.Y.html 'Size: '
// @hasraw - '1 byte'
// @!hasraw - ' bytes'
pub struct Y(u8);

// @hasraw type_layout/struct.Z.html 'Size: '
// @hasraw - '0 bytes'
pub struct Z;

// We can't compute layout for generic types.
// @hasraw type_layout/struct.Generic.html 'Unable to compute type layout, possibly due to this type having generic parameters'
// @!hasraw - 'Size: '
pub struct Generic<T>(T);

// We *can*, however, compute layout for types that are only generic over lifetimes,
// because lifetimes are a type-system construct.
// @hasraw type_layout/struct.GenericLifetimes.html 'Size: '
// @hasraw - ' bytes'
pub struct GenericLifetimes<'a>(&'a str);

// @hasraw type_layout/struct.Unsized.html 'Size: '
// @hasraw - '(unsized)'
pub struct Unsized([u8]);

// @hasraw type_layout/type.TypeAlias.html 'Size: '
// @hasraw - ' bytes'
pub type TypeAlias = X;

// @hasraw type_layout/type.GenericTypeAlias.html 'Size: '
// @hasraw - '8 bytes'
pub type GenericTypeAlias = (Generic<(u32, ())>, Generic<u32>);

// Regression test for the rustdoc equivalent of #85103.
// @hasraw type_layout/type.Edges.html 'Encountered an error during type layout; the type failed to be normalized.'
pub type Edges<'a, E> = std::borrow::Cow<'a, [E]>;

// @!hasraw type_layout/trait.MyTrait.html 'Size: '
pub trait MyTrait {}

// @hasraw type_layout/enum.Variants.html 'Size: '
// @hasraw - '2 bytes'
// @hasraw - '<code>A</code>: 0 bytes'
// @hasraw - '<code>B</code>: 1 byte'
pub enum Variants {
    A,
    B(u8),
}

// @hasraw type_layout/enum.WithNiche.html 'Size: '
// @has - //p '4 bytes'
// @hasraw - '<code>None</code>: 0 bytes'
// @hasraw - '<code>Some</code>: 4 bytes'
pub enum WithNiche {
    None,
    Some(std::num::NonZeroU32),
}

// @hasraw type_layout/enum.Uninhabited.html 'Size: '
// @hasraw - '0 bytes (<a href="https://doc.rust-lang.org/stable/reference/glossary.html#uninhabited">uninhabited</a>)'
pub enum Uninhabited {}

// @hasraw type_layout/struct.Uninhabited2.html 'Size: '
// @hasraw - '8 bytes (<a href="https://doc.rust-lang.org/stable/reference/glossary.html#uninhabited">uninhabited</a>)'
pub struct Uninhabited2(std::convert::Infallible, u64);
