struct S;

fn main() {
    let b = [0; S];
    //~^ ERROR mismatched types
    //~| expected `usize`, found `S`
}
