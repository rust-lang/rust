// This test checks that rustdoc is combining `<code>` tags as expected.

#![crate_name = "foo"]

//@ has 'foo/index.html'

// First we check that summary lines also get the `<code>` merge.
//@ has - '//dd/code' 'Stream<Item = io::Result<Bytes>>'
//@ !has - '//dd/code/code' 'Stream<Item = io::Result<Bytes>>'
// Then we check that the docblocks have it too.
//@ has 'foo/struct.Foo.html'
//@ has - '//*[@class="docblock"]//code' 'Stream<Item = io::Result<Bytes>>'
//@ has - '//*[@class="docblock"]//code' 'Stream<Item = io::Result<Bytes>>'
/// [`Stream`](crate::Foo)`<Item = `[`io::Result`](crate::Foo)`<`[`Bytes`](crate::Foo)`>>`
pub struct Foo;

impl Foo {
    //@ has - '//*[@class="impl-items"]//*[@class="docblock"]//code' '<Stream>'
    /// A `<`[`Stream`](crate::Foo)`>` stuff.
    pub fn bar() {}

    //@ has - '//*[@class="impl-items"]//*[@class="docblock"]//code' '<'
    //@ has - '//*[@class="impl-items"]//*[@class="docblock"]//a' 'Stream a'
    //@ has - '//*[@class="impl-items"]//*[@class="docblock"]//code' 'Stream'
    //@ has - '//*[@class="impl-items"]//*[@class="docblock"]//code' '>'
    /// A `<`[`Stream` a](crate::Foo)`>` stuff.
    pub fn foo() {}
}
