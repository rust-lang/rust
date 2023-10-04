// ignore-windows
// compile-flags: --no-defaults

#![crate_name="issue_26995"]

// @has src/issue_26995/dev/null.html
// @has issue_26995/null/index.html '//a/@href' '../../src/issue_26995/dev/null.html'
#[path="/dev/null"]
pub mod null;
