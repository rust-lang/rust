// aux-build:issue-40001-plugin.rs
// ignore-stage1
// compile-flags:-Z attr_tool=plugin

#![feature(plugin)]
#![plugin(issue_40001_plugin)]

#[plugin::whitelisted_attr]
fn main() {}
