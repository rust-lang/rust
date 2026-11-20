// Regression test for <https://github.com/rust-lang/rust/issues/32077>.

#![crate_name = "foo"]

pub struct GenericStruct<T>(T);

impl<T> GenericStruct<T> {
    pub fn on_gen(arg: T) {}
}

impl GenericStruct<u32> {
    pub fn on_u32(arg: u32) {}
}

pub trait Foo {}
pub trait Bar {}

impl<T> Foo for GenericStruct<T> {}
impl Bar for GenericStruct<u32> {}

//@ has 'foo/type.TypedefStruct.html'
// We check that "Aliased Type" is also present as a title in the sidebar.
//@ has - '//*[@class="sidebar-elems"]//h3/a[@href="#aliased-type"]' 'Aliased Type'
// We check that we have the implementation of the type alias itself.
//@ has - '//*[@id="impl-GenericStruct%3Cu8%3E"]/h3' 'impl TypedefStruct'
//@ has - '//*[@id="method.on_alias"]/h4' 'pub fn on_alias()'
// This trait implementation doesn't match the type alias parameters so shouldn't appear in docs.
//@ !has - '//h3' 'impl Bar for GenericStruct<u32> {}'
// Same goes for the `Deref` impl.
//@ !has - '//h2' 'Methods from Deref<Target = u32>'
//@ count - '//nav[@class="sidebar"]//a' 'on_alias' 1
//@ !has - '//nav[@class="sidebar"]//a' 'on_gen'
//@ !has - '//nav[@class="sidebar"]//a' 'Foo'
//@ !has - '//nav[@class="sidebar"]//a' 'Bar'
//@ !has - '//nav[@class="sidebar"]//a' 'on_u32'
// TypedefStruct inlined to GenericStruct
//@ hasraw 'type.impl/foo/struct.GenericStruct.js' 'TypedefStruct'
//@ hasraw 'type.impl/foo/struct.GenericStruct.js' 'method.on_gen'
//@ hasraw 'type.impl/foo/struct.GenericStruct.js' 'Foo'
pub type TypedefStruct = GenericStruct<u8>;

impl TypedefStruct {
    pub fn on_alias() {}
}

impl std::ops::Deref for GenericStruct<u32> {
    type Target = u32;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct Wrap<T>(GenericStruct<T>);

//@ has 'foo/type.Alias.html'
//@ !has - '//h2' 'Methods from Deref<Target = u32>'
//@ !has - '//*[@id="impl-Deref-for-Wrap%3CT%3E"]/h3' 'impl<T> Deref for Wrap<T>'
//@ hasraw 'type.impl/foo/struct.Wrap.js' 'impl-Deref-for-Wrap%3CT%3E'
// Deref Methods aren't gathered for type aliases, though the actual impl is.
//@ !hasraw 'type.impl/foo/struct.Wrap.js' 'BITS'
pub type Alias = Wrap<u32>;

impl<T> std::ops::Deref for Wrap<T> {
    type Target = GenericStruct<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
