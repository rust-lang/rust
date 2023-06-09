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

// @has 'foo/type.TypedefStruct.html'
// We check that we have the implementation of the type alias itself.
// @has - '//*[@id="impl-TypedefStruct"]/h3' 'impl TypedefStruct'
// @has - '//*[@id="method.on_alias"]/h4' 'pub fn on_alias()'
// @has - '//*[@id="impl-GenericStruct%3CT%3E"]/h3' 'impl<T> GenericStruct<T>'
// @has - '//*[@id="method.on_gen"]/h4' 'pub fn on_gen(arg: T)'
// @has - '//*[@id="impl-Foo-for-GenericStruct%3CT%3E"]/h3' 'impl<T> Foo for GenericStruct<T>'
// This trait implementation doesn't match the type alias parameters so shouldn't appear in docs.
// @!has - '//h3' 'impl Bar for GenericStruct<u32> {}'
// Same goes for the `Deref` impl.
// @!has - '//h2' 'Methods from Deref<Target = u32>'
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

// @has 'foo/type.Alias.html'
// @has - '//h2' 'Methods from Deref<Target = u32>'
// @has - '//*[@id="impl-Deref-for-Wrap%3CT%3E"]/h3' 'impl<T> Deref for Wrap<T>'
pub type Alias = Wrap<u32>;

impl<T> std::ops::Deref for Wrap<T> {
    type Target = GenericStruct<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
