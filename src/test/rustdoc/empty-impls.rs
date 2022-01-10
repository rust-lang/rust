#![crate_name = "foo"]

// @has foo/struct.Foo.html
// @has - '//div[@id="synthetic-implementations-list"]/div[@id="impl-Send-for-Foo"]' 'impl Send for Foo'
pub struct Foo;

pub trait EmptyTrait {}

// @has - '//div[@id="trait-implementations-list"]/div[@id="impl-EmptyTrait-for-Foo"]' 'impl EmptyTrait for Foo'
impl EmptyTrait for Foo {}

pub trait NotEmpty {
    fn foo(&self);
}

// @has - '//div[@id="trait-implementations-list"]/details/summary/div[@id="impl-NotEmpty-for-Foo"]' 'impl NotEmpty for Foo'
impl NotEmpty for Foo {
    fn foo(&self) {}
}
