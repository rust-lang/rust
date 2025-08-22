pub(crate) mod escape;
pub(crate) mod format;
pub(crate) mod highlight;
pub(crate) mod layout;
mod length_limit;
pub(crate) mod macro_expansion;
// used by the error-index generator, so it needs to be public
pub mod markdown;
pub(crate) mod render;
pub(crate) mod sources;
pub(crate) mod static_files;
pub(crate) mod toc;
mod url_parts_builder;

#[cfg(test)]
mod tests;
