// This test is to ensure that the raw underscore lifetime won't emit two duplicate errors.
// See issue #143152

//@ edition: 2021

fn f<'r#_>(){}
//~^ ERROR `_` cannot be a raw lifetime
//~| ERROR `'_` cannot be used here [E0637]

fn main() {}
