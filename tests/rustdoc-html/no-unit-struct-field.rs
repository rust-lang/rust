// This test ensures that the tuple struct fields are not generated in the
// search index.

// vlqhex encoding ` = 0, a = 1, e = 5
//@ !hasraw search.index/name/*.js 'a0'
//@ !hasraw search.index/name/*.js 'a1'
//@ hasraw search.index/name/*.js 'efoo_a'
//@ hasraw search.index/name/*.js 'ebar_a'

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
