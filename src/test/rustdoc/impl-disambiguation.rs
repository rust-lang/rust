#![crate_name = "foo"]

pub trait Foo {}

pub struct Bar<T> { field: T }

// @has foo/trait.Foo.html '//*[@class="item-list"]//code' \
//     "impl Foo for Bar<u8>"
impl Foo for Bar<u8> {}
// @has foo/trait.Foo.html '//*[@class="item-list"]//code' \
//     "impl Foo for Bar<u16>"
impl Foo for Bar<u16> {}
// @has foo/trait.Foo.html '//*[@class="item-list"]//code' \
//     "impl<'a> Foo for &'a Bar<u8>"
impl<'a> Foo for &'a Bar<u8> {}

pub mod mod1 {
    pub struct Baz {}
}

pub mod mod2 {
    pub enum Baz {}
}

// @has foo/trait.Foo.html '//*[@class="item-list"]//code' \
//     "impl Foo for foo::mod1::Baz"
impl Foo for mod1::Baz {}
// @has foo/trait.Foo.html '//*[@class="item-list"]//code' \
//     "impl<'a> Foo for &'a foo::mod2::Baz"
impl<'a> Foo for &'a mod2::Baz {}
