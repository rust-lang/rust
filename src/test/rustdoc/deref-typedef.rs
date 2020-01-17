#![crate_name = "foo"]

// @has 'foo/struct.Bar.html'
// @has '-' '//*[@id="deref-methods"]' 'Methods from Deref<Target = FooC>'
// @has '-' '//*[@class="impl-items"]//*[@id="method.foo_a"]' 'pub fn foo_a(&self)'
// @has '-' '//*[@class="impl-items"]//*[@id="method.foo_b"]' 'pub fn foo_b(&self)'
// @has '-' '//*[@class="impl-items"]//*[@id="method.foo_c"]' 'pub fn foo_c(&self)'
// @has '-' '//*[@class="sidebar-title"]' 'Methods from Deref<Target=FooC>'
// @has '-' '//*[@class="sidebar-links"]/a[@href="#method.foo_a"]' 'foo_a'
// @has '-' '//*[@class="sidebar-links"]/a[@href="#method.foo_b"]' 'foo_b'
// @has '-' '//*[@class="sidebar-links"]/a[@href="#method.foo_c"]' 'foo_c'

pub struct FooA;
pub type FooB = FooA;
pub type FooC = FooB;

impl FooA {
    pub fn foo_a(&self) {}
}

impl FooB {
    pub fn foo_b(&self) {}
}

impl FooC {
    pub fn foo_c(&self) {}
}

pub struct Bar;
impl std::ops::Deref for Bar {
    type Target = FooC;
    fn deref(&self) -> &Self::Target { unimplemented!() }
}
