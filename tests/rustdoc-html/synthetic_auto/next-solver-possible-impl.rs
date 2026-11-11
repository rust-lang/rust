//@ compile-flags: -Znext-solver=globally

#![feature(auto_traits)]
#![crate_name = "foo"]

pub auto trait Marker {}

//@ has 'foo/struct.MyType.html'
//@ !has - '//*[@id="synthetic-implementations-list"]//*[@class="impl"]' 'Marker for MyType<T>'
pub struct MyType<T>(T);

impl Marker for MyType<u32> {}
