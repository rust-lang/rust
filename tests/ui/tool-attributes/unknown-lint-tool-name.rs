#![deny(foo::bar)] //~ ERROR unknown tool name `foo` found in scoped lint: `foo::bar`
                   //~| ERROR unknown tool name `foo` found in scoped lint: `foo::bar`
                   //~| ERROR unknown tool name `foo` found in scoped lint: `foo::bar`

#[allow(foo::bar)] //~ ERROR unknown tool name `foo` found in scoped lint: `foo::bar`
                   //~| ERROR unknown tool name `foo` found in scoped lint: `foo::bar`
                   //~| ERROR unknown tool name `foo` found in scoped lint: `foo::bar`
fn main() {}
