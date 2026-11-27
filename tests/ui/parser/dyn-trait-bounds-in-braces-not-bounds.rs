//@ edition:2021

fn body_after_dyn() -> dyn { 0 }
//~^ ERROR at least one trait is required for an object type

fn comma_in_braces(_: &dyn {Send, Sync}) {}
//~^ ERROR expected parameter name, found `{`
//~| ERROR expected one of `!`, `(`, `)`, `,`, `?`, `[`, `async`, `const`, `for`, `use`, `~`, lifetime, or path, found `{`
//~| ERROR at least one trait is required for an object type

impl dyn {}
//~^ ERROR at least one trait is required for an object type

fn main() {}
