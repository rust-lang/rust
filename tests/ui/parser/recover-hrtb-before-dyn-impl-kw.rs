trait Trait {}

fn test(_: &for<'a> dyn Trait) {}
//~^ ERROR `for<...>` expected after `dyn`, not before

fn test2(_: for<'a> impl Trait) {}
//~^ ERROR `for<...>` expected after `impl`, not before

fn main() {}
