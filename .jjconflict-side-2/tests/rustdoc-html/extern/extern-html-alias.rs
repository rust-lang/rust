//@ compile-flags:-Z unstable-options --extern-html-root-url externs_name=https://renamed.example.com  --extern-html-root-url empty=https://bad.invalid
//@ aux-crate:externs_name=empty.rs
//@ edition: 2018

extern crate externs_name as renamed;

//@ has extern_html_alias/index.html
//@ has - '//a/@href' 'https://renamed.example.com/empty/index.html'
pub use renamed as yet_different_name;
