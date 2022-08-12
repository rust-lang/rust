#![feature(trait_alias)]

pub trait SelfInput = Fn(&mut Self);

pub fn f(_f: &dyn SelfInput) {}
//~^ ERROR the trait alias `SelfInput` cannot be made into an object [E0038]

fn main() {}
