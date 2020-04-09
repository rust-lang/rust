// aux-build:lint-group-plugin-test.rs
// compile-flags: -F unused -A unused

fn main() {
    let x = 1;
    //~^ ERROR unused variable: `x`
}
