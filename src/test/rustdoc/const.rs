#![crate_type="lib"]

#![feature(const_fn)]

pub struct Foo;

impl Foo {
    // @has const/struct.Foo.html '//*[@id="new.v"]//code' 'const unsafe fn new'
    pub const unsafe fn new() -> Foo {
        Foo
    }
}
