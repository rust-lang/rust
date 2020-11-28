#![crate_name = "foo"]

#![feature(doc_keyword)]

// @has foo/index.html '//h2[@id="keywords"]' 'Keywords'
// @has foo/index.html '//a[@href="keyword.match.html"]' 'match'
// @has foo/index.html '//div[@class="block items"]//a/@href' '#keywords'
// @has foo/keyword.match.html '//a[@class="keyword"]' 'match'
// @has foo/keyword.match.html '//span[@class="in-band"]' 'Keyword match'
// @has foo/keyword.match.html '//section[@id="main"]//div[@class="docblock"]//p' 'this is a test!'
// @has foo/index.html '//a/@href' '../foo/index.html'
// @!has foo/foo/index.html
// @!has-dir foo/foo
#[doc(keyword = "match")]
/// this is a test!
mod foo{}

// @has foo/keyword.foo.html '//section[@id="main"]//div[@class="docblock"]//p' 'hello'
#[doc(keyword = "foo")]
/// hello
mod bar {}
