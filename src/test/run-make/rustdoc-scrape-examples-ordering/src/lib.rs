// @has foobar/fn.ok.html '//*[@class="docblock scraped-example-list"]' 'ex2'
// @has foobar/fn.ok.html '//*[@class="more-scraped-examples"]' 'ex1'
// @has foobar/fn.ok.html '//*[@class="highlight focus"]' ''
// @has foobar/fn.ok.html '//*[@class="highlight"]' ''

pub fn ok(_x: i32) {}
