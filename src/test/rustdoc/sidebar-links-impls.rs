// issue #56018: "Implementations on Foreign Types" sidebar items should link to specific impls

#![crate_name = "foo"]

// @has foo/trait.Foo.html
// @has - '//*[@class="sidebar-title"]/a[@href="#foreign-impls"]' 'Implementations on Foreign Types'
// @has - '//h2[@id="foreign-impls"]' 'Implementations on Foreign Types'
// @has - '//*[@class="sidebar-links"]/a[@href="#impl-Foo-for-u32"]' 'u32'
// @has - '//div[@id="impl-Foo-for-u32"]//h3[@class="code-header in-band"]' 'impl Foo for u32'
// @has - '//*[@class="sidebar-links"]/a[@href="#impl-Foo-for-%26%27a%20str"]' "&'a str"
// @has - '//div[@id="impl-Foo-for-%26%27a%20str"]//h3[@class="code-header in-band"]' "impl<'a> Foo for &'a str"
pub trait Foo {}

impl Foo for u32 {}

impl<'a> Foo for &'a str {}

// Issue #91118" "Implementors column on trait page is always empty"

pub struct B;
pub struct A;
struct C;

// @has foo/trait.Bar.html
// @has - '//*[@class="sidebar-title"]/a[@href="#implementors"]' 'Implementors'
// @has - '//h2[@id="implementors"]' 'Implementors'
// @!has - '//*[@class="sidebar-title"]/a[@href="#foreign-impls"]'
// @!has - '//h2[@id="foreign-impls"]'

// @has - '//*[@class="sidebar-links"]/a[@href="#impl-Bar"]' '&A'
// @has - '//*[@class="sidebar-links"]/a[@href="#impl-Bar-1"]' 'A'
// @has - '//*[@class="sidebar-links"]/a[@href="#impl-Bar-2"]' 'B'

// @has - '//div[@id="impl-Bar"]//h3[@class="code-header in-band"]' 'impl Bar for &A'
// @has - '//div[@id="impl-Bar-1"]//h3[@class="code-header in-band"]' 'impl Bar for A'
// @has - '//div[@id="impl-Bar-2"]//h3[@class="code-header in-band"]' 'impl Bar for B'
pub trait Bar {}
impl Bar for B {}
impl Bar for A {}
impl Bar for &A {}
impl Bar for C {}

// @has foo/trait.NotImpled.html
// @!has - '//*[@class="sidebar-title"]/a[@href="#implementors"]'
// FIXME: Is this the semantics we want
// @has - '//h2[@id="implementors"]' 'Implementors'
// @!has - '//*[@class="sidebar-title"]/a[@href="#foreign-impls"]'
// @!has - '//h2[@id="foreign-impls"]'
pub trait NotImpled {}
