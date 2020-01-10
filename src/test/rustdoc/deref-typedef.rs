#![crate_name = "foo"]

// @has 'foo/struct.Bar.html'
// @has '-' '//*[@id="deref-methods"]' 'Methods from Deref<Target = FooB>'
// @has '-' '//*[@class="impl-items"]//*[@id="method.happy"]' 'pub fn happy(&self)'
// @has '-' '//*[@class="sidebar-title"]' 'Methods from Deref<Target=FooB>'
// @has '-' '//*[@class="sidebar-links"]/a[@href="#method.happy"]' 'happy'
pub struct FooA;
pub type FooB = FooA;

impl FooA {
    pub fn happy(&self) {}
}

pub struct Bar;
impl std::ops::Deref for Bar {
    type Target = FooB;
    fn deref(&self) -> &FooB { unimplemented!() }
}
