// https://github.com/rust-lang/rust/issues/118180
#![crate_name="foo"]

//@ has foo/type.Enum.html
pub enum Enum {
    //@ has - '//*[@id="variant.Empty"]//h3' 'Empty()'
    Empty(),
}

//@ has foo/type.Empty.html
//@ has - '//pre/code' 'Empty()'
pub struct Empty();
