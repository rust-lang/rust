#![deny(broken_intra_doc_links)]
pub enum Foo {
    Bar {
        abc: i32,
        /// [Self::Bar::abc]
        xyz: i32,
    },
}
