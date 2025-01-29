// https://github.com/rust-lang/rust/issues/82209

#![crate_name = "foo"]
#![deny(rustdoc::broken_intra_doc_links)]
pub enum Foo {
    Bar {
        abc: i32,
        /// [Self::Bar::abc]
        xyz: i32,
    },
}

//@ has foo/enum.Foo.html '//a/@href' 'enum.Foo.html#variant.Bar.field.abc'
