// compile-flags: --show-type-layout -Z unstable-options

// @hastext type_layout/struct.Foo.html 'Size: '
// @hastext - ' bytes'
// @has - '//*[@id="layout"]/a[@href="#layout"]' ''
pub struct Foo {
    pub a: usize,
    b: Vec<String>,
}

// @hastext type_layout/enum.Bar.html 'Size: '
// @hastext - ' bytes'
pub enum Bar<'a> {
    A(String),
    B(&'a str, (std::collections::HashMap<String, usize>, Foo)),
}

// @hastext type_layout/union.Baz.html 'Size: '
// @hastext - ' bytes'
pub union Baz {
    a: &'static str,
    b: usize,
    c: &'static [u8],
}

// @hastext type_layout/struct.X.html 'Size: '
// @hastext - ' bytes'
pub struct X(usize);

// @hastext type_layout/struct.Y.html 'Size: '
// @hastext - '1 byte'
// @!has - ' bytes'
pub struct Y(u8);

// @hastext type_layout/struct.Z.html 'Size: '
// @hastext - '0 bytes'
pub struct Z;

// We can't compute layout for generic types.
// @hastext type_layout/struct.Generic.html 'Unable to compute type layout, possibly due to this type having generic parameters'
// @!has - 'Size: '
pub struct Generic<T>(T);

// We *can*, however, compute layout for types that are only generic over lifetimes,
// because lifetimes are a type-system construct.
// @hastext type_layout/struct.GenericLifetimes.html 'Size: '
// @hastext - ' bytes'
pub struct GenericLifetimes<'a>(&'a str);

// @hastext type_layout/struct.Unsized.html 'Size: '
// @hastext - '(unsized)'
pub struct Unsized([u8]);

// @hastext type_layout/type.TypeAlias.html 'Size: '
// @hastext - ' bytes'
pub type TypeAlias = X;

// @hastext type_layout/type.GenericTypeAlias.html 'Size: '
// @hastext - '8 bytes'
pub type GenericTypeAlias = (Generic<(u32, ())>, Generic<u32>);

// Regression test for the rustdoc equivalent of #85103.
// @hastext type_layout/type.Edges.html 'Encountered an error during type layout; the type failed to be normalized.'
pub type Edges<'a, E> = std::borrow::Cow<'a, [E]>;

// @!has type_layout/trait.MyTrait.html 'Size: '
pub trait MyTrait {}

// @hastext type_layout/enum.Variants.html 'Size: '
// @hastext - '2 bytes'
// @hastext - '<code>A</code>: 0 bytes'
// @hastext - '<code>B</code>: 1 byte'
pub enum Variants {
    A,
    B(u8),
}

// @hastext type_layout/enum.WithNiche.html 'Size: '
// @has - //p '4 bytes'
// @hastext - '<code>None</code>: 0 bytes'
// @hastext - '<code>Some</code>: 4 bytes'
pub enum WithNiche {
    None,
    Some(std::num::NonZeroU32),
}
