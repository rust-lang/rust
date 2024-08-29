// This test enforces that the (renamed) reexports are present in the search results.
// This is a DWIM case, since renaming the export probably means the intent is also different.
// For the de-duplication case of exactly the same name, see reexport-dedup

pub mod fmt {
    pub struct Subscriber;
}
mod foo {
    pub struct AnotherOne;
}

pub use fmt::Subscriber as FmtSubscriber;
pub use foo::AnotherOne;
