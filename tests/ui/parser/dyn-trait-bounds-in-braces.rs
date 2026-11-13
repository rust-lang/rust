//@ edition:2021

fn auto_traits(_: &dyn {Send + Sync}) {}
//~^ ERROR incorrect braces around trait bounds

fn single_bound(_: &dyn {Send}) {}
//~^ ERROR incorrect braces around trait bounds

fn with_lifetime(_: Box<dyn {Send + 'static}>) {}
//~^ ERROR incorrect braces around trait bounds

fn main() {}
