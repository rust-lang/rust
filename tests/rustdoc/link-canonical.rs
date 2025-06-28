#![crate_name = "foo"]
#![feature(doc_link_canonical)]
#![doc(html_link_canonical = "https://foo.example/")]

//@ has 'foo/index.html'
//@ has - '//head/link[@rel="canonical"][@href="https://foo.example/foo/index.html"]' ''

//@ has 'foo/struct.FooBaz.html'
//@ has - '//head/link[@rel="canonical"][@href="https://foo.example/foo/struct.FooBaz.html"]' ''
pub struct FooBaz;
