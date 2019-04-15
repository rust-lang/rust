#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
enum E {
    V { field: bool }
}
fn test_E(x: E) {
    let field = true;
    if x == E::V { field } {}
    //~^ ERROR expected value, found struct variant `E::V`
    //~| ERROR mismatched types
    let y: usize = ();
    //~^ ERROR mismatched types
}

fn main() {}
