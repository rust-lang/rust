// compile-flags: --show-type-layout -Z unstable-options

// @has type_layout/struct.Foo.html 'Size: '
// @has - ' bytes'
pub struct Foo {
    pub a: usize,
    b: Vec<String>,
}

// @has type_layout/enum.Bar.html 'Size: '
// @has - ' bytes'
pub enum Bar<'a> {
    A(String),
    B(&'a str, (std::collections::HashMap<String, usize>, Foo)),
}

// @has type_layout/union.Baz.html 'Size: '
// @has - ' bytes'
pub union Baz {
    a: &'static str,
    b: usize,
    c: &'static [u8],
}

// @has type_layout/struct.X.html 'Size: '
// @has - ' bytes'
pub struct X(usize);

// @has type_layout/struct.Y.html 'Size: '
// @has - '1 byte'
// @!has - ' bytes'
pub struct Y(u8);

// @has type_layout/struct.Z.html 'Size: '
// @has - '0 bytes'
pub struct Z;

// We can't compute layout for generic types.
// @has type_layout/struct.Generic.html 'Unable to compute type layout, possibly due to this type having generic parameters'
// @!has - 'Size: '
pub struct Generic<T>(T);

// We *can*, however, compute layout for types that are only generic over lifetimes,
// because lifetimes are a type-system construct.
// @has type_layout/struct.GenericLifetimes.html 'Size: '
// @has - ' bytes'
pub struct GenericLifetimes<'a>(&'a str);

// @has type_layout/struct.Unsized.html 'Size: '
// @has - '(unsized)'
pub struct Unsized([u8]);

// @!has type_layout/trait.MyTrait.html 'Size: '
pub trait MyTrait {}

// @has type_layout/enum.Variants.html 'Size: '
// @has - '2 bytes'
// @has - '<code>A</code>: 0 bytes'
// @has - '<code>B</code>: 1 byte'
pub enum Variants {
    A,
    B(u8),
}

// @has type_layout/enum.WithNiche.html 'Size: '
// @has - //p '4 bytes'
// @has - '<code>None</code>: 0 bytes'
// @has - '<code>Some</code>: 4 bytes'
pub enum WithNiche {
    None,
    Some(std::num::NonZeroU32),
}
