#![forbid(non_snake_case)]

#[allow(non_snake_case)]
//~^ ERROR allow(non_snake_case) overruled by outer forbid(non_snake_case)
fn main() {
}
