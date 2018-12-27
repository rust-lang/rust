#![deny(foo::bar)] //~ ERROR an unknown tool name found in scoped lint: `foo::bar`

#[allow(foo::bar)] //~ ERROR an unknown tool name found in scoped lint: `foo::bar`
fn main() {}
