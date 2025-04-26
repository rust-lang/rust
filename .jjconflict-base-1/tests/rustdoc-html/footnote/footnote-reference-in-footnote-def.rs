// Checks that footnote references in footnote definitions are correctly generated.
// Regression test for <https://github.com/rust-lang/rust/issues/131946>.

#![crate_name = "foo"]

//@ has 'foo/index.html'
//@ has - '//*[@class="docblock"]/p/sup[@id="fnref1"]/a[@href="#fn1"]' '1'
//@ has - '//li[@id="fn1"]/p' 'meow'
//@ has - '//li[@id="fn1"]/p/sup[@id="fnref2"]/a[@href="#fn2"]' '2'
//@ has - '//li[@id="fn1"]//a[@href="#fn2"]' '2'
//@ has - '//li[@id="fn2"]/p' 'uwu'
//@ has - '//li[@id="fn2"]/p/sup[@id="fnref1-2"]/a[@href="#fn1"]' '1'
//@ has - '//li[@id="fn2"]//a[@href="#fn1"]' '1'

//! # footnote-hell
//!
//! Hello [^a].
//!
//! [^a]: meow [^b]
//! [^b]: uwu [^a]
