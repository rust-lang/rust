#![crate_name = "foo"]

//@ has 'foo/struct.Foo.html'
//@ has - '//*[@id="impl-Send-for-Foo"]' 'impl !Send for Foo'
//@ has - '//*[@id="impl-Sync-for-Foo"]' 'impl !Sync for Foo'
pub struct Foo(*const i8);
pub trait Whatever: Send {}
impl<T: Send + ?Sized> Whatever for T {}
