// @has foobar/fn.ok.html '//*[@class="docblock scraped-example-list"]' 'ex2'
// @has foobar/fn.ok.html '//*[@class="more-scraped-examples"]' 'ex1'
// @has foobar/fn.ok.html '//*[@class="highlight focus"]' '1'
// @has foobar/fn.ok.html '//*[@class="highlight"]' '2'
// @has foobar/fn.ok.html '//*[@class="highlight focus"]' '0'

pub fn ok(_x: i32) {}
