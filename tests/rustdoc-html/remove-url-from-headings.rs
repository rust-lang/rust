// It actually checks that the link is kept in the headings as expected now.

#![crate_name = "foo"]

//@ has foo/fn.foo.html
//@ has - '//a[@href="http://a.a"]' 'stuff'
//@ has - '//*[@id="implementing-stuff-somewhere"]' 'Implementing stuff somewhere'
//@ has - '//a[@href="http://b.b"]' 'one'
//@ has - '//*[@id="another-one-urg"]' 'Another one urg'

/// fooo
///
/// # Implementing [stuff](http://a.a "title") somewhere
///
/// hello
///
/// # Another [one][two] urg
///
/// [two]: http://b.b
pub fn foo() {}
