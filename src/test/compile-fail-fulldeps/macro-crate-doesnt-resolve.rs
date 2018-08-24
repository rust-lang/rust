// aux-build:macro_crate_test.rs

#[macro_use] #[no_link]
extern crate macro_crate_test;

fn main() {
    macro_crate_test::foo(); //~ ERROR cannot find function `foo` in module `macro_crate_test`
}
