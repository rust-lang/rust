#![crate_name = "foo"]

// @has 'foo/struct.Bar.html'
// @has '-' '//*[@id="deref-methods-FooJ"]' 'Methods from Deref<Target = FooJ>'
// @has '-' '//*[@class="impl-items"]//*[@id="method.foo_a"]' 'pub fn foo_a(&self)'
// @has '-' '//*[@class="impl-items"]//*[@id="method.foo_b"]' 'pub fn foo_b(&self)'
// @has '-' '//*[@class="impl-items"]//*[@id="method.foo_c"]' 'pub fn foo_c(&self)'
// @has '-' '//*[@class="impl-items"]//*[@id="method.foo_j"]' 'pub fn foo_j(&self)'
// @has '-' '//*[@class="sidebar-title"]/a[@href="#deref-methods-FooJ"]' 'Methods from Deref<Target=FooJ>'
// @has '-' '//*[@class="sidebar-links"]/a[@href="#method.foo_a"]' 'foo_a'
// @has '-' '//*[@class="sidebar-links"]/a[@href="#method.foo_b"]' 'foo_b'
// @has '-' '//*[@class="sidebar-links"]/a[@href="#method.foo_c"]' 'foo_c'
// @has '-' '//*[@class="sidebar-links"]/a[@href="#method.foo_j"]' 'foo_j'

pub struct FooA;
pub type FooB = FooA;
pub type FooC = FooB;
pub type FooD = FooC;
pub type FooE = FooD;
pub type FooF = FooE;
pub type FooG = FooF;
pub type FooH = FooG;
pub type FooI = FooH;
pub type FooJ = FooI;

impl FooA {
    pub fn foo_a(&self) {}
}

impl FooB {
    pub fn foo_b(&self) {}
}

impl FooC {
    pub fn foo_c(&self) {}
}

impl FooJ {
    pub fn foo_j(&self) {}
}

pub struct Bar;
impl std::ops::Deref for Bar {
    type Target = FooJ;
    fn deref(&self) -> &Self::Target { unimplemented!() }
}
