#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
enum E {
    V { field: bool },
    I { field1: bool, field2: usize },
    J { field: isize },
    K { field: &'static str},
}
fn test_E(x: E) {
    let field = true;
    if x == E::V { field } {}
    //~^ ERROR expected value, found struct variant `E::V`
    //~| ERROR mismatched types
    if x == E::I { field1: true, field2: 42 } {}
    //~^ ERROR struct literals are not allowed here
    if x == E::V { field: false } {}
    //~^ ERROR struct literals are not allowed here
    if x == E::J { field: -42 } {}
    //~^ ERROR struct literals are not allowed here
    if x == E::K { field: "" } {}
    //~^ ERROR struct literals are not allowed here
    let y: usize = ();
    //~^ ERROR mismatched types
}

fn main() {}
