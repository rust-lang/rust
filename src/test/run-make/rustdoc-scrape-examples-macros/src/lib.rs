// Scraped example should only include line numbers for items b and c in ex.rs
// @!has foobar/fn.f.html '//*[@class="line-numbers"]' '14'
// @has foobar/fn.f.html '//*[@class="line-numbers"]' '15'
// @has foobar/fn.f.html '//*[@class="line-numbers"]' '21'
// @!has foobar/fn.f.html '//*[@class="line-numbers"]' '22'

pub fn f() {}

#[macro_export]
macro_rules! a_rules_macro {
  ($e:expr) => { ($e, foobar::f()); }
}
