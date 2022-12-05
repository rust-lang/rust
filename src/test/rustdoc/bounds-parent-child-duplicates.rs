// This test ensures that if a child item's bounds are duplicated with the parent, they are not
// generated in the documentation.

#![crate_name = "foo"]

pub trait Bar {}
pub trait Bar2 {}
pub trait Bar3 {}
pub trait Bar4 {}

// @has 'foo/trait.Foo.html'
pub trait Foo<'a, T: Bar + 'a> where T: Bar2 {
    // @has - '//*[@id="method.foo"]/h4' 'fn foo()'
    // `Bar` shouldn't appear in the bounds.
    // @!has - '//*[@id="method.foo"]/h4' 'Bar'
    fn foo() where T: Bar {}
    // @has - '//*[@id="method.foo2"]/h4' 'fn foo2()'
    // `Bar2` shouldn't appear in the bounds.
    // @!has - '//*[@id="method.foo2"]/h4' 'Bar2'
    fn foo2() where T: Bar2 {}
    // @has - '//*[@id="method.foo3"]/h4' "fn foo3<'b>()where T: Bar3, 'a: 'b,"
    fn foo3<'b>() where T: Bar3, 'a: 'b {}
    // @has - '//*[@id="method.foo4"]/h4' "fn foo4()where T: Bar3,"
    fn foo4() where T: Bar2 + Bar3 {}
}

pub struct X;

// @has 'foo/struct.X.html'
impl<'a, T: Bar> X where T: Bar2 {
    // @has - '//*[@id="method.foo"]/h4' 'fn foo()'
    // @!has - '//*[@id="method.foo"]/h4' 'Bar'
    // `Bar` shouldn't appear in the bounds.
    pub fn foo() where T: Bar {}
    // @has - '//*[@id="method.foo2"]/h4' 'fn foo2()'
    // `Bar2` shouldn't appear in the bounds.
    // @!has - '//*[@id="method.foo2"]/h4' 'Bar2'
    pub fn foo2() where T: Bar2 {}
    // @has - '//*[@id="method.foo3"]/h4' "fn foo3<'b>()where T: Bar3, 'a: 'b,"
    pub fn foo3<'b>() where 'a: 'b, T: Bar3 {}
    // @has - '//*[@id="method.foo4"]/h4' "fn foo4()where T: Bar3,"
    pub fn foo4() where T: Bar2 + Bar3 {}
}
