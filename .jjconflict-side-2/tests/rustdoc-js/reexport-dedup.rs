// This test enforces that the (renamed) reexports are present in the search results.
#![crate_name = "foo"]

pub mod fmt {
    pub struct Subscriber;
}
mod foo {
    pub struct AnotherOne;
}

pub use fmt::Subscriber;
pub use foo::AnotherOne;
