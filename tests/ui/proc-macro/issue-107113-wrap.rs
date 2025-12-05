//@ edition:2021
//@ proc-macro: issue-107113.rs
//@ ignore-backends: gcc

#[macro_use]
extern crate issue_107113;

#[issue_107113::main] //~ ERROR mismatched types [E0308]
async fn main() -> std::io::Result<()> {}
