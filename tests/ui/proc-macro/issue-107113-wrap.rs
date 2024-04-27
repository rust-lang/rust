//@ edition:2021
//@ aux-build:issue-107113.rs

#[macro_use]
extern crate issue_107113;

#[issue_107113::main] //~ ERROR mismatched types [E0308]
async fn main() -> std::io::Result<()> {}
