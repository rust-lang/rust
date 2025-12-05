#![crate_name = "foo"]

#![feature(rustdoc_internals)]

//@ has foo/index.html '//h2[@id="keywords"]' 'Keywords'
//@ has foo/index.html '//a[@href="keyword.match.html"]' 'match'
//@ has foo/index.html '//div[@class="sidebar-elems"]//li/a' 'Keywords'
//@ has foo/index.html '//div[@class="sidebar-elems"]//li/a/@href' '#keywords'
//@ has foo/keyword.match.html '//h1' 'Keyword match'
//@ has foo/keyword.match.html '//section[@id="main-content"]//div[@class="docblock"]//p' 'this is a test!'
//@ has foo/index.html '//a/@href' '../foo/index.html'
//@ !has foo/foo/index.html
//@ !has-dir foo/foo
//@ !has foo/index.html '//span' '🔒'
#[doc(keyword = "match")]
/// this is a test!
mod foo{}

//@ has foo/keyword.break.html '//section[@id="main-content"]//div[@class="docblock"]//p' 'hello'
#[doc(keyword = "break")]
/// hello
mod bar {}
