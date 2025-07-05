//@ compile-flags:-Z unstable-options --extern-html-root-url extern_prelude_name=https://renamed.example.com  --extern-html-root-url empty=https://bad.invalid
//@ aux-crate:extern_prelude_name=empty.rs

extern crate extern_prelude_name as internal_ident_name;

//@ has extern_html_alias/index.html
//@ has - '//a/@href' 'https://renamed.example.com/empty/index.html'
pub use internal_ident_name as yet_different_name;
