// This test ensures that no footnote reference is generated inside
// summary doc.

#![crate_name = "foo"]

// @has 'foo/index.html'
// @has - '//*[@class="desc docblock-short"]' 'hello bla'
// @!has - '//*[@class="desc docblock-short"]/sup' '1'

// @has 'foo/struct.S.html'
// @has - '//*[@class="docblock"]//sup' '1'
// @has - '//*[@class="docblock"]' 'hello 1 bla'

/// hello [^foot] bla
///
/// [^foot]: blabla
pub struct S;
