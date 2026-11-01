// https://github.com/rust-lang/rust/issues/108925
#![crate_name="foo"]

//@ has foo/enum.MyThing.html
//@ has - '//code' 'Shown'
//@ !has - '//code' 'NotShown'
//@ !has - '//code' '// some variants omitted'
#[non_exhaustive]
pub enum MyThing {
    Shown,
    #[doc(hidden)]
    NotShown,
}
