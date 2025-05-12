// Scraped example should only include line numbers for items b and c in ex.rs
//@ !has foobar/fn.f.html '//span[@data-nosnippet]' '14'
//@ has foobar/fn.f.html '//span[@data-nosnippet]' '15'
//@ has foobar/fn.f.html '//span[@data-nosnippet]' '21'
//@ !has foobar/fn.f.html '//span[@data-nosnippet]' '22'

pub fn f() {}

#[macro_export]
macro_rules! a_rules_macro {
    ($e:expr) => {
        ($e, foobar::f());
    };
}
