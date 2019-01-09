// aux-build:macro_crate_test.rs

#[macro_use] #[no_link]
extern crate macro_crate_test;

fn main() {
    unexported_macro!();
    //~^ ERROR cannot find macro `unexported_macro!` in this scope
}
