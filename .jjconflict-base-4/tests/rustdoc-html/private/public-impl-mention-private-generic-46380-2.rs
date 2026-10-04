// https://github.com/rust-lang/rust/issues/46380
#![crate_name="foo"]

pub trait PublicTrait<T> {}

//@ has foo/struct.PublicStruct.html
pub struct PublicStruct;

//@ !has - '//*[@class="impl"]' 'impl PublicTrait<PrivateStruct> for PublicStruct'
impl PublicTrait<PrivateStruct> for PublicStruct {}

struct PrivateStruct;
