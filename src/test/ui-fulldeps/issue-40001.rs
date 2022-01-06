// run-pass
// aux-build:issue-40001-plugin.rs
// ignore-stage1

#![feature(plugin)]
#![cfg_attr(bootstrap, feature(register_tool))]
#![plugin(issue_40001_plugin)] //~ WARNING compiler plugins are deprecated
#![register_tool(plugin)]

#[plugin::allowed_attr]
fn main() {}
