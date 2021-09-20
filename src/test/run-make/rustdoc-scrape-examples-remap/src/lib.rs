// @has foobar/b/fn.foo.html '//*[@class="scraped-example"]' 'ex.rs'
// @has foobar/c/fn.foo.html '//*[@class="scraped-example"]' 'ex.rs'

#[path = "a.rs"]
pub mod b;

#[path = "a.rs"]
pub mod c;
