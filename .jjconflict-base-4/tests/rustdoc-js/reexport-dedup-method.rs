// This test enforces that the (renamed) reexports are present in the search results.
#![crate_name = "foo"]

pub mod fmt {
    pub struct Subscriber;
    impl Subscriber {
        pub fn dostuff(&self) {}
    }
}
mod foo {
    pub struct AnotherOne;
    impl AnotherOne {
        pub fn dostuff(&self) {}
    }
}

pub use fmt::Subscriber;
pub use foo::AnotherOne;
