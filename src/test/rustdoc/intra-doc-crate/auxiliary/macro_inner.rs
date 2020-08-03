#![crate_name = "macro_inner"]
#![deny(broken_intra_doc_links)]

pub struct Foo;

/// See also [`Foo`]
#[macro_export]
macro_rules! my_macro {
    () => {}
}
