// Regression test for <https://github.com/rust-lang/rust/issues/148184>.
// It ensures that the macro expansion correctly handles its "class stack".

//@ compile-flags: -Zunstable-options --generate-macro-expansion
//@ aux-build:one-line-expand.rs

#![crate_name = "foo"]

extern crate just_some_proc;

//@ has 'src/foo/one-line-expand.rs.html'
//@ has - '//*[@class="comment"]' '//'
//@ has - '//*[@class="original"]' '#[just_some_proc::repro]'
//@ has - '//*[@class="original"]/*[@class="attr"]' '#[just_some_proc::repro]'
//@ has - '//code/*[@class="kw"]' 'struct '

//
#[just_some_proc::repro]
struct Repro;
