#![forbid(deprecated)]

#[allow(deprecated)]
//~^ ERROR allow(deprecated) incompatible
//~| ERROR allow(deprecated) incompatible
//~| ERROR allow(deprecated) incompatible
fn main() {
}
