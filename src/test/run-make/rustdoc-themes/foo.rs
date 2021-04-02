// @has test.css
// @has foo/struct.Foo.html
// @has - '//link[@rel="stylesheet"]/@href' '../test.css'
pub struct Foo;
