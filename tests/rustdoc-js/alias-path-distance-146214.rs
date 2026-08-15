#![crate_name = "alias_path_distance"]

pub struct Foo;
pub struct Bar;

impl Foo {
    #[doc(alias = "zzz")]
    pub fn baz() {}
}

impl Bar {
    #[doc(alias = "zzz")]
    pub fn baz() {}
}
