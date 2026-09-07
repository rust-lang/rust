// This test ensures that no footnote reference is generated inside
// summary doc.

#![crate_name = "foo"]

//@ has 'foo/index.html'
//@ has - '//dd' 'hello bla'
//@ !has - '//dd/sup' '1'

//@ has 'foo/struct.S.html'
//@ has - '//*[@class="docblock"]//sup' '1'
//@ has - '//*[@class="docblock"]' 'hello 1 bla'

/// hello [^foot] bla
///
/// [^foot]: blabla
pub struct S;
