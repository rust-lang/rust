#![crate_name = "foo"]

#![feature(associated_const_equality)]
pub enum ParseMode {
    Raw,
}
pub trait Parse {
    const PARSE_MODE: ParseMode;
}
pub trait RenderRaw {}

// @hasraw foo/trait.RenderRaw.html 'impl'
// @hasraw foo/trait.RenderRaw.html 'ParseMode::Raw'
impl<T: Parse<PARSE_MODE = { ParseMode::Raw }>> RenderRaw for T {}
