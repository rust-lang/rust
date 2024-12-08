// @has foobar/fn.ok.html '//*[@class="docblock scraped-example-list"]//*[@class="prev"]' ''
// @has foobar/fn.ok.html '//*[@class="more-scraped-examples"]' ''
// @has src/ex/ex.rs.html
// @has foobar/fn.ok.html '//a[@href="../src/ex/ex.rs.html#2"]' ''

pub fn ok() {}
