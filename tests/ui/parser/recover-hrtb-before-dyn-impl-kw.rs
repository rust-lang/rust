//@ edition: 2015

trait Trait {}

fn test(_: &for<'a> dyn Trait) {}
//~^ ERROR `for<...>` expected after `dyn`, not before

fn test2(_: for<'a> impl Trait) {}
//~^ ERROR `for<...>` expected after `impl`, not before

// Issue #118564
type A2 = dyn<for<> dyn>;
//~^ ERROR expected identifier, found `>`

fn main() {}
