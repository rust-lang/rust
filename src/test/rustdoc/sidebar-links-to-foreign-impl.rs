// issue #56018: "Implementations on Foreign Types" sidebar items should link to specific impls

#![crate_name = "foo"]

// @has foo/trait.Foo.html
// @has - '//*[@class="sidebar-title"][@href="#foreign-impls"]' 'Implementations on Foreign Types'
// @has - '//h2[@id="foreign-impls"]' 'Implementations on Foreign Types'
// @has - '//*[@class="sidebar-links"]/a[@href="#impl-Foo-for-u32"]' 'u32'
// @has - '//div[@id="impl-Foo-for-u32"]//code' 'impl Foo for u32'
// @has - '//*[@class="sidebar-links"]/a[@href="#impl-Foo-for-%26%27a%20str"]' "&'a str"
// @has - '//div[@id="impl-Foo-for-%26%27a%20str"]//code' "impl<'a> Foo for &'a str"
pub trait Foo {}

impl Foo for u32 {}

impl<'a> Foo for &'a str {}
