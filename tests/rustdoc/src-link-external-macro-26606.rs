//@ aux-build:issue-26606-macro.rs
//@ ignore-cross-compile
//@ build-aux-docs

// https://github.com/rust-lang/rust/issues/26606
#![crate_name="issue_26606"]

//@ has issue_26606_macro/macro.make_item.html
#[macro_use]
extern crate issue_26606_macro;

//@ has issue_26606/constant.FOO.html
//@ has - '//a[@href="../src/issue_26606/src-link-external-macro-26606.rs.html#14"]' 'Source'
make_item!(FOO);
