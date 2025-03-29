//@ check-pass

#[cfg(false)]
auto trait Foo {}
//~^ WARN `auto` traits are unstable
//~| WARN unstable syntax can change at any point in the future, causing a hard error!

fn main() {}
