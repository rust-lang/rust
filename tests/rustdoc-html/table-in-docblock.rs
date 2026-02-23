#![crate_name = "foo"]

//@ has foo/struct.Foo.html
//@ count - '//*[@class="docblock"]/div/table' 2
//@ !has - '//*[@class="docblock"]/table' ''
/// | hello | hello2 |
/// | ----- | ------ |
/// | data  | data2  |
pub struct Foo;

impl Foo {
    /// | hello | hello2 |
    /// | ----- | ------ |
    /// | data  | data2  |
    pub fn foo(&self) {}
}
