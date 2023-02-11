// This test enforces that the (renamed) reexports are present in the search results.

pub mod fmt {
    pub struct Subscriber;
}
mod foo {
    pub struct AnotherOne;
}

pub use foo::AnotherOne;
pub use fmt::Subscriber as FmtSubscriber;
