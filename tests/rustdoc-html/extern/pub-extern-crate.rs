//@ aux-build:pub-extern-crate.rs

//@ has pub_extern_crate/index.html
//@ !has - '//code' 'pub extern crate inner'
//@ has - '//a/@href' 'inner/index.html'
//@ has pub_extern_crate/inner/index.html
//@ has pub_extern_crate/inner/struct.SomeStruct.html
#[doc(inline)]
pub extern crate inner;
