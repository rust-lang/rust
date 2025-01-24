// issue #56018: "Implementations on Foreign Types" sidebar items should link to specific impls

#![crate_name = "foo"]

//@ has foo/trait.Foo.html
//@ has - '//div[@class="sidebar-elems"]//h3/a[@href="#foreign-impls"]' 'Implementations on Foreign Types'
//@ has - '//h2[@id="foreign-impls"]' 'Implementations on Foreign Types'
//@ has - '//*[@class="sidebar-elems"]//section//a[@href="#impl-Foo-for-u32"]' 'u32'
//@ has - '//*[@id="impl-Foo-for-u32"]//h3[@class="code-header"]' 'impl Foo for u32'
//@ has - "//*[@class=\"sidebar-elems\"]//section//a[@href=\"#impl-Foo-for-%26str\"]" "&'a str"
//@ has - "//*[@id=\"impl-Foo-for-%26str\"]//h3[@class=\"code-header\"]" "impl<'a> Foo for &'a str"
pub trait Foo {}

impl Foo for u32 {}

impl<'a> Foo for &'a str {}
