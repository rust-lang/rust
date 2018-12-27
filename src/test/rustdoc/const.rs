#![crate_type="lib"]

pub struct Foo;

impl Foo {
    // @has const/struct.Foo.html '//code[@id="new.v"]' 'const unsafe fn new'
    pub const unsafe fn new() -> Foo {
        Foo
    }
}
