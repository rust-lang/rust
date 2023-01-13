#![crate_name = "foo"]

pub enum Enum{Variant}
pub use self::Enum::Variant;

// @!has foo/index.html '//a/@href' './Enum/index.html'
