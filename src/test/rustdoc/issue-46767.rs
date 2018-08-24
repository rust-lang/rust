#![crate_name = "foo"]

mod private {
    pub enum Enum{Variant}
}
pub use self::private::Enum::*;

// @!has foo/index.html '//a/@href' './private/index.html'
