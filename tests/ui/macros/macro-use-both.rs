// run-pass
// aux-build:two_macros.rs

#[macro_use(macro_one, macro_two)]
extern crate two_macros;

pub fn main() {
    macro_one!();
    macro_two!();
}
