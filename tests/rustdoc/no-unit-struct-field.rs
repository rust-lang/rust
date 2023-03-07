// This test ensures that the tuple struct fields are not generated in the
// search index.

// @!hasraw search-index.js '"0"'
// @!hasraw search-index.js '"1"'
// @hasraw search-index.js '"foo_a"'
// @hasraw search-index.js '"bar_a"'

pub struct Bar(pub u32, pub u8);
pub struct Foo {
    pub foo_a: u8,
}
pub enum Enum {
    Foo(u8),
    Bar {
        bar_a: u8,
    },
}
