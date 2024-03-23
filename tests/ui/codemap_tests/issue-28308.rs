//@revisions: edition2024 edition2021
//@[edition2024] edition:2024
//@[edition2024] compile-flags: -Z unstable-options
//@[edition2021] edition:2021
fn main() {
    assert!("foo");
    //[edition2024]~^ ERROR mismatched types
    //[edition2021]~^^ ERROR cannot apply unary operator `!` to type `&'static str`
}
