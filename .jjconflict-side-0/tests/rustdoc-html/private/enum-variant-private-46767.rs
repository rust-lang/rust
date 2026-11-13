// https://github.com/rust-lang/rust/issues/46767
#![crate_name = "foo"]

mod private {
    pub enum Enum{Variant}
}
pub use self::private::Enum::*;

//@ !has-dir foo/private
//@ !has foo/index.html '//a/@href' 'private/index.html'
