// @has structfields/struct.Foo.html
pub struct Foo {
    // @has - //pre "pub a: ()"
    pub a: (),
    // @has - //pre "/* private fields */"
    // @!has - //pre "b: ()"
    b: (),
    // @!has - //pre "c: usize"
    #[doc(hidden)]
    c: usize,
    // @has - //pre "pub d: usize"
    pub d: usize,
}

// @has structfields/struct.Bar.html
pub struct Bar {
    // @has - //pre "pub a: ()"
    pub a: (),
    // @!has - //pre "/* private fields */"
}

// @has structfields/enum.Qux.html
pub enum Qux {
    Quz {
        // @has - //pre "a: ()"
        a: (),
        // @!has - //pre "b: ()"
        #[doc(hidden)]
        b: (),
        // @has - //pre "c: usize"
        c: usize,
        // @has - //pre "/* private fields */"
    },
}

// @has structfields/struct.Baz.html //pre "pub struct Baz { /* private fields */ }"
pub struct Baz {
    x: u8,
    #[doc(hidden)]
    pub y: u8,
}

// @has structfields/struct.Quux.html //pre "pub struct Quux {}"
pub struct Quux {}
