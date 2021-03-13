#![crate_name = "foo"]
#![deny(broken_intra_doc_links)]
pub enum Foo {
    Bar {
        abc: i32,
        /// [Self::Bar::abc]
        xyz: i32,
    },
}

// @has foo/enum.Foo.html '//a/@href' '../foo/enum.Foo.html#variant.Bar.field.abc'
