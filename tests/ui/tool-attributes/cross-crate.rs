//@ aux-build: use_tool.rs

// `use_tool` references tool "foo", and we want to check that it has no impact on this crate.
extern crate use_tool;

#[foo::bar] //~ ERROR cannot find module or crate `foo` in this scope
#[allow(foo::baz)] //~ ERROR unknown tool name `foo`
                   //~| ERROR unknown tool name `foo`
                   //~| ERROR unknown tool name `foo`
fn main() {}
