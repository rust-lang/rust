//@ known-bug: #132980
// Originally a rustdoc test. Should be moved back there with @has checks
// readded once fixed.
// Previous issue (before mgca): https://github.com/rust-lang/rust/issues/105952
#![crate_name = "foo"]
#![feature(associated_const_equality, min_generic_const_args)]
pub enum ParseMode {
    Raw,
}
pub trait Parse {
    #[type_const]
    const PARSE_MODE: ParseMode;
}
pub trait RenderRaw {}

impl<T: Parse<PARSE_MODE = { ParseMode::Raw }>> RenderRaw for T {}
