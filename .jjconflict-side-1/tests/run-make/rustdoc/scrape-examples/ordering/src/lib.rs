//@ has foobar/fn.ok.html '//*[@class="docblock scraped-example-list"]' 'ex2'
//@ has foobar/fn.ok.html '//*[@class="more-scraped-examples"]' 'ex1'
//@ has foobar/fn.ok.html '//*[@class="highlight focus"]' 'ok'
//@ has foobar/fn.ok.html '//*[@class="highlight"]' 'ok'

pub fn ok(_x: i32) {}
