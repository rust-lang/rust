#![crate_name = "foo"]

// @has foo/fn.foo.html '//a[@class="test-arrow"]/@href' 'https://play.rust-lang.org/?code=%23!%5Ballow(unused)%5D%0Afn%20main()%20%7B%0Alet%20x%20%3D%2012%3B%0A%7D&edition=2015'
/// ```
/// let x = 12;
/// ```
pub fn foo() {}
