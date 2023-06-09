// @has foobar/b/fn.foo.html '//*[@class="scraped-example expanded"]' 'ex.rs'
// @has foobar/c/fn.foo.html '//*[@class="scraped-example expanded"]' 'ex.rs'

#[path = "a.rs"]
pub mod b;

#[path = "a.rs"]
pub mod c;
