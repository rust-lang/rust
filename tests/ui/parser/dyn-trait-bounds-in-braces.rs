//@ edition:2021

fn auto_traits(_: &dyn {Send + Sync}) {}
//~^ ERROR expected parameter name, found `{`
//~| ERROR expected one of `!`, `(`, `)`, `,`, `?`, `[`, `async`, `const`, `for`, `use`, `~`, lifetime, or path, found `{`
//~| ERROR at least one trait is required for an object type

fn single_bound(_: &dyn {Send}) {}
//~^ ERROR expected parameter name, found `{`
//~| ERROR expected one of `!`, `(`, `)`, `,`, `?`, `[`, `async`, `const`, `for`, `use`, `~`, lifetime, or path, found `{`
//~| ERROR at least one trait is required for an object type

fn with_lifetime(_: Box<dyn {Send + 'static}>) {}
//~^ ERROR expected one of `!`, `(`, `,`, `:`, `=`, `>`, `?`, `[`, `async`, `const`, `for`, `use`, `~`, lifetime, or path, found `{`

fn main() {}
