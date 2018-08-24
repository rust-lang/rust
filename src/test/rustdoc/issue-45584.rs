#![crate_name = "foo"]

pub trait Bar<T, U> {}

// @has 'foo/struct.Foo1.html'
pub struct Foo1;
// @count - '//*[@id="implementations-list"]/*[@class="impl"]' 1
// @has - '//*[@class="impl"]' "impl Bar<Foo1, &'static Foo1> for Foo1"
impl Bar<Foo1, &'static Foo1> for Foo1 {}

// @has 'foo/struct.Foo2.html'
pub struct Foo2;
// @count - '//*[@id="implementations-list"]/*[@class="impl"]' 1
// @has - '//*[@class="impl"]' "impl Bar<&'static Foo2, Foo2> for u8"
impl Bar<&'static Foo2, Foo2> for u8 {}
