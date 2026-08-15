//@ compile-flags:-Z unstable-options --extern-html-root-url yet_another_name=https://bad.invalid --extern-html-root-url renamed_privately=https://bad.invalid --extern-html-root-url renamed_locally=https://bad.invalid --extern-html-root-url empty=https://localhost
//@ aux-crate:externs_name=empty.rs
//@ edition: 2018

mod m {
    pub extern crate externs_name as renamed_privately;
}

// renaming within the crate's source code is not supposed to affect CLI flags
extern crate externs_name as renamed_locally;

//@ has extern_html_fallback/index.html
//@ has - '//a/@href' 'https://localhost/empty/index.html'
pub use crate::renamed_locally as yet_another_name;
