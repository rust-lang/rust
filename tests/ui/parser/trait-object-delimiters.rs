// edition:2018

fn foo1(_: &dyn Drop + AsRef<str>) {} //~ ERROR ambiguous `+` in a type
//~^ ERROR only auto traits can be used as additional traits in a trait object

fn foo2(_: &dyn (Drop + AsRef<str>)) {} //~ ERROR incorrect braces around trait bounds

fn foo2_no_space(_: &dyn(Drop + AsRef<str>)) {} //~ ERROR incorrect braces around trait bounds

fn foo3(_: &dyn {Drop + AsRef<str>}) {} //~ ERROR expected parameter name, found `{`
//~^ ERROR expected one of `!`, `(`, `)`, `*`, `,`, `?`, `for`, `~`, lifetime, or path, found `{`
//~| ERROR at least one trait is required for an object type

fn foo4(_: &dyn <Drop + AsRef<str>>) {} //~ ERROR expected identifier, found `<`

fn foo5(_: &(dyn Drop + dyn AsRef<str>)) {} //~ ERROR invalid `dyn` keyword
//~^ ERROR only auto traits can be used as additional traits in a trait object

fn main() {}
