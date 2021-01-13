#![forbid(non_snake_case)]

#[allow(non_snake_case)]
//~^ ERROR allow(non_snake_case) incompatible
//~| ERROR allow(non_snake_case) incompatible
//~| ERROR allow(non_snake_case) incompatible
fn main() {
}
