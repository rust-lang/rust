// This test ensures that multiple references to a single footnote and
// corresponding back links work as expected.

#![crate_name = "foo"]

//@ has 'foo/index.html'
//@ has - '//*[@class="docblock"]/p/sup[@id="fnref1"]/a[@href="#fn1"]' '1'
//@ has - '//*[@class="docblock"]/p/sup[@id="fnref2"]/a[@href="#fn2"]' '2'
//@ has - '//*[@class="docblock"]/p/sup[@id="fnref2-2"]/a[@href="#fn2"]' '2'
//@ has - '//li[@id="fn1"]/p' 'meow'
//@ has - '//li[@id="fn1"]/p/a[@href="#fnref1"]' '↩'
//@ has - '//li[@id="fn2"]/p' 'uwu'
//@ has - '//li[@id="fn2"]/p/a[@href="#fnref2"]/sup' '1'
//@ has - '//li[@id="fn2"]/p/sup/a[@href="#fnref2-2"]' '2'

//! # Footnote, references and back links
//!
//! Single: [^a].
//!
//! Double: [^b] [^b].
//!
//! [^a]: meow
//! [^b]: uwu
