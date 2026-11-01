// Regression test for <https://github.com/rust-lang/rust/issues/96335>

fn main() {
    0.....{loop{}1};
    //~^ ERROR unexpected token
    //~| ERROR mismatched types
}
