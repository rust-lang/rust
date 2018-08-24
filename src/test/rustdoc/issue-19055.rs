// @has issue_19055/trait.Any.html
pub trait Any {}

impl<'any> Any + 'any {
    // @has - '//*[@id="method.is"]' 'fn is'
    pub fn is<T: 'static>(&self) -> bool { loop {} }

    // @has - '//*[@id="method.downcast_ref"]' 'fn downcast_ref'
    pub fn downcast_ref<T: 'static>(&self) -> Option<&T> { loop {} }

    // @has - '//*[@id="method.downcast_mut"]' 'fn downcast_mut'
    pub fn downcast_mut<T: 'static>(&mut self) -> Option<&mut T> { loop {} }
}

pub trait Foo {
    fn foo(&self) {}
}

// @has - '//*[@id="method.foo"]' 'fn foo'
impl Foo for Any {}
