//@ edition:2015

fn path_named_dyn(_: &dyn {Send + Sync}) {}
//~^ ERROR expected parameter name, found `{`
//~| ERROR expected one of `!`, `(`, `)`, `,`, `::`, or `<`, found `{`
//~| ERROR cannot find type `dyn` in this scope

fn main() {}
