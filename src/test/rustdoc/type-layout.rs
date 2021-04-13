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
// @has - ' byte'
// @!has - ' bytes'
pub struct Y(u8);

// @!has type_layout/struct.Generic.html 'Size: '
pub struct Generic<T>(T);

// @has type_layout/struct.Unsized.html 'Size: '
// @has - '(unsized)'
pub struct Unsized([u8]);

// @!has type_layout/type.TypeAlias.html 'Size: '
pub type TypeAlias = X;

// @!has type_layout/trait.MyTrait.html 'Size: '
pub trait MyTrait {}
