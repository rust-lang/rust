#![crate_name = "foo"]

// @has foo/struct.Foo.html
// @has - '//div[@id="synthetic-implementations-list"]/h3[@id="impl-Send"]' 'impl Send for Foo'
pub struct Foo;

pub trait EmptyTrait {}

// @has - '//div[@id="trait-implementations-list"]/h3[@id="impl-EmptyTrait"]' 'impl EmptyTrait for Foo'
impl EmptyTrait for Foo {}

pub trait NotEmpty {
    fn foo(&self);
}

// @has - '//div[@id="trait-implementations-list"]/details/summary/h3[@id="impl-NotEmpty"]' 'impl NotEmpty for Foo'
impl NotEmpty for Foo {
    fn foo(&self) {}
}
