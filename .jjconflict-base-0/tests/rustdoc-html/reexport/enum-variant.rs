// This test ensures that reexported enum variants correctly link to the original variant.

#![crate_name = "foo"]

pub enum Foo {
    S {
        x: u32,
    },
}

//@ has 'foo/index.html'

//@ has - '//*[@class="item-table reexports"]/*[@id="reexport.S"]//a[@href="enum.Foo.html#variant.S"]' 'S'
pub use self::Foo::S;
