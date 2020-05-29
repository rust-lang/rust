// aux-build:issue-26606-macro.rs
// ignore-cross-compile
// build-aux-docs

// @has issue_26606_macro/macro.make_item.html
#[macro_use]
extern crate issue_26606_macro;

// @has issue_26606/constant.FOO.html
// @has - '//a/@href' '../src/issue_26606_macro/issue-26606-macro.rs.html#3'
make_item!(FOO);
